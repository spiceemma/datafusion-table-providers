use std::{fmt, sync::Arc};

use arrow::datatypes::SchemaRef;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::TableReference,
    common::{not_impl_err, Constraints, SchemaExt},
    datasource::{
        sink::{DataSink, DataSinkExec},
        TableProvider, TableType,
    },
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::{dml::InsertOp, Expr},
    physical_plan::{metrics::MetricsSet, DisplayAs, DisplayFormatType, ExecutionPlan},
};
use futures::StreamExt;
use snafu::prelude::*;

use crate::util::{
    constraints::{self},
    count_exec::make_count_exec,
    dml::{
        assignments_to_sql, delete_statement_returning_count, filters_to_sql, update_statement,
        DeletionExec, DeletionSink, UpdateExec, UpdateSink,
    },
    on_conflict::OnConflict,
    retriable_error::check_and_mark_retriable_error,
};

use crate::postgres::Postgres;
use crate::sql::sql_provider_datafusion::expr::Engine;

use super::to_datafusion_error;

/// The `WHERE` condition of a `DELETE` or `UPDATE`, or `None` for every row.
///
/// Rendered for `PostgreSQL`, not the generic engine: the generic rendering of a timestamp
/// literal is `TO_TIMESTAMP(<whole seconds>)`, which drops the fraction of a second (and
/// rounds a time before the epoch toward it), so the statement selects different rows than
/// the condition did.
fn where_clause(filters: &[Expr]) -> datafusion::error::Result<Option<String>> {
    if filters.is_empty() {
        return Ok(None);
    }
    filters_to_sql(filters, Some(Engine::Postgres)).map(Some)
}

/// The `SET` list of an `UPDATE`, rendered for `PostgreSQL` for the reason [`where_clause`]
/// gives: the generic rendering writes a timestamp without its fraction of a second.
fn set_clause(assignments: &[(String, Expr)]) -> datafusion::error::Result<String> {
    assignments_to_sql(assignments, Some(Engine::Postgres))
}

#[derive(Debug, Clone)]
pub struct PostgresTableWriter {
    pub read_provider: Arc<dyn TableProvider>,
    postgres: Arc<Postgres>,
    on_conflict: Option<OnConflict>,
}

impl PostgresTableWriter {
    pub fn create(
        read_provider: Arc<dyn TableProvider>,
        postgres: Postgres,
        on_conflict: Option<OnConflict>,
    ) -> Arc<Self> {
        Arc::new(Self {
            read_provider,
            postgres: Arc::new(postgres),
            on_conflict,
        })
    }

    pub fn postgres(&self) -> Arc<Postgres> {
        Arc::clone(&self.postgres)
    }
}

#[async_trait]
impl TableProvider for PostgresTableWriter {
    fn schema(&self) -> SchemaRef {
        self.read_provider.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(self.postgres.constraints())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        self.read_provider
            .scan(state, projection, filters, limit)
            .await
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        op: InsertOp,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // `InsertOp::Replace` promises an atomic per-row upsert (no delete leg).
        // Postgres delivers that only with an `ON CONFLICT (...) DO UPDATE`,
        // i.e. `OnConflict::Upsert`. A missing `on_conflict` (plain append) or a
        // `DoNothing`/`DoNothingAll` (leaves the existing row unchanged) does
        // NOT replace, so refuse rather than silently accept a non-replacing
        // write.
        if matches!(op, InsertOp::Replace)
            && !matches!(self.on_conflict, Some(OnConflict::Upsert(_)))
        {
            return not_impl_err!(
                "InsertOp::Replace requires an on_conflict upsert target on the Postgres writer"
            );
        }
        Ok(Arc::new(DataSinkExec::new(
            input,
            Arc::new(PostgresDataSink::new(
                Arc::clone(&self.postgres),
                op,
                self.on_conflict.clone(),
                self.schema(),
            )),
            None,
        )) as _)
    }

    async fn delete_from(
        &self,
        _state: &dyn Session,
        filters: Vec<Expr>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let sql_where = where_clause(&filters)?;
        let table = self.postgres.table_reference().clone();
        let postgres = self.postgres();

        Ok(Arc::new(DeletionExec::new(Arc::new(
            PostgresDeletionSink {
                postgres,
                table,
                sql_where,
            },
        ))))
    }

    async fn update(
        &self,
        _state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if assignments.is_empty() {
            return make_count_exec(0);
        }

        let set_clause = set_clause(&assignments)?;
        let postgres = self.postgres();

        let sql_where = where_clause(&filters)?;
        let sql = update_statement(
            self.postgres.table_reference(),
            &set_clause,
            sql_where.as_deref(),
        );

        Ok(Arc::new(UpdateExec::new(Arc::new(PostgresUpdateSink {
            postgres,
            sql,
        }))))
    }
}

struct PostgresDeletionSink {
    postgres: Arc<Postgres>,
    table: TableReference,
    sql_where: Option<String>,
}

#[async_trait]
impl DeletionSink for PostgresDeletionSink {
    #[expect(clippy::cast_sign_loss)]
    async fn delete_from(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut db_conn = self.postgres.connect().await?;
        let pg_conn = Postgres::postgres_conn(&mut db_conn)?;
        let tx = pg_conn.conn.transaction().await?;

        let sql = delete_statement_returning_count(&self.table, self.sql_where.as_deref());
        let row = tx.query_one(&sql, &[]).await?;
        let deleted: i64 = row.get(0);
        tx.commit().await?;

        Ok(deleted as u64)
    }
}

struct PostgresUpdateSink {
    postgres: Arc<Postgres>,
    sql: String,
}

#[async_trait]
impl UpdateSink for PostgresUpdateSink {
    async fn execute_update(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut db_conn = self.postgres.connect().await?;
        let pg_conn = Postgres::postgres_conn(&mut db_conn)?;
        let tx = pg_conn.conn.transaction().await?;

        let count = tx.execute(&self.sql, &[]).await?;
        tx.commit().await?;

        Ok(count)
    }
}

#[derive(Clone)]
struct PostgresDataSink {
    postgres: Arc<Postgres>,
    overwrite: InsertOp,
    on_conflict: Option<OnConflict>,
    schema: SchemaRef,
}

#[async_trait]
impl DataSink for PostgresDataSink {
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> datafusion::common::Result<u64> {
        let mut num_rows = 0;

        let mut db_conn = self.postgres.connect().await.map_err(to_datafusion_error)?;
        let postgres_conn = Postgres::postgres_conn(&mut db_conn).map_err(to_datafusion_error)?;

        let tx = postgres_conn
            .conn
            .transaction()
            .await
            .context(super::UnableToBeginTransactionSnafu)
            .map_err(to_datafusion_error)?;

        if matches!(self.overwrite, InsertOp::Overwrite) {
            self.postgres
                .delete_all_table_data(&tx)
                .await
                .map_err(to_datafusion_error)?;
        }

        let postgres_fields = self
            .postgres
            .schema
            .fields
            .iter()
            .map(|f| {
                Arc::new(Field::new(
                    f.name(),
                    if f.data_type() == &DataType::LargeUtf8 {
                        DataType::Utf8
                    } else {
                        f.data_type().clone()
                    },
                    f.is_nullable(),
                ))
            })
            .collect::<Vec<_>>();

        let postgres_schema = Arc::new(Schema::new(postgres_fields));

        while let Some(batch) = data.next().await {
            let batch = batch.map_err(check_and_mark_retriable_error)?;

            // for the purposes of PostgreSQL, LargeUtf8 is equivalent to Utf8
            // because Postgres physically cannot store anything larger than 1Gb in text (VARCHAR)
            // normalize LargeUtf8 fields to Utf8 for both the incoming batch, and Postgres if it happens to specify any
            let batch_fields = batch
                .schema_ref()
                .fields()
                .iter()
                .map(|f| {
                    Arc::new(Field::new(
                        f.name(),
                        if f.data_type() == &DataType::LargeUtf8 {
                            DataType::Utf8
                        } else {
                            f.data_type().clone()
                        },
                        f.is_nullable(),
                    ))
                })
                .collect::<Vec<_>>();
            let batch_schema = Arc::new(Schema::new(batch_fields));

            if !Arc::clone(&postgres_schema).equivalent_names_and_types(&batch_schema) {
                return Err(to_datafusion_error(super::Error::SchemaValidationError {
                    table_name: self.postgres.table.to_string(),
                }));
            }

            let batch_num_rows = batch.num_rows();

            if batch_num_rows == 0 {
                continue;
            };

            num_rows += batch_num_rows as u64;

            constraints::validate_batch_with_constraints(
                vec![batch.clone()],
                self.postgres.constraints(),
                &crate::util::constraints::UpsertOptions::default(),
            )
            .await
            .context(super::ConstraintViolationSnafu)
            .map_err(to_datafusion_error)?;

            self.postgres
                .insert_batch(&tx, batch, self.on_conflict.clone())
                .await
                .map_err(to_datafusion_error)?;
        }

        tx.commit()
            .await
            .context(super::UnableToCommitPostgresTransactionSnafu)
            .map_err(to_datafusion_error)?;

        Ok(num_rows)
    }
}

impl PostgresDataSink {
    fn new(
        postgres: Arc<Postgres>,
        overwrite: InsertOp,
        on_conflict: Option<OnConflict>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            postgres,
            overwrite,
            on_conflict,
            schema,
        }
    }
}

impl std::fmt::Debug for PostgresDataSink {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PostgresDataSink")
    }
}

impl DisplayAs for PostgresDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "PostgresDataSink")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::{prelude::col, scalar::ScalarValue};

    fn naive_nanos(nanos: i64) -> Expr {
        Expr::Literal(ScalarValue::TimestampNanosecond(Some(nanos), None), None)
    }

    /// 2024-06-01 00:00:00.5 and 1969-12-31 23:59:59.5 keep their fractions, and a naive
    /// literal is read in UTC rather than in the session's time zone.
    #[test]
    fn a_where_clause_keeps_a_timestamps_fraction_of_a_second() {
        let filters = [
            col("ts").lt(naive_nanos(1_717_200_000_500_000_000)),
            col("ts").gt_eq(naive_nanos(-500_000_000)),
        ];
        assert_eq!(
            where_clause(&filters).expect("where clause").as_deref(),
            Some(
                r#"("ts" < TO_TIMESTAMP(1717200000.5) AT TIME ZONE 'UTC') AND ("ts" >= TO_TIMESTAMP(-0.5) AT TIME ZONE 'UTC')"#
            )
        );
        assert_eq!(where_clause(&[]).expect("no filters"), None);
    }

    /// 2025-02-03 04:05:06.789 is written with its milliseconds.
    #[test]
    fn a_set_clause_keeps_a_timestamps_fraction_of_a_second() {
        let assignments = [("ts".to_string(), naive_nanos(1_738_555_506_789_000_000))];
        assert_eq!(
            set_clause(&assignments).expect("set clause"),
            r#""ts" = TO_TIMESTAMP(1738555506.789) AT TIME ZONE 'UTC'"#
        );
    }
}
