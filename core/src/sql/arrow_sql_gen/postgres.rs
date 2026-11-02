use std::convert;
use std::io::Read;
use std::sync::Arc;

use crate::sql::arrow_sql_gen::arrow::map_data_type_to_array_builder_optional;
use crate::sql::arrow_sql_gen::statement::map_data_type_to_column_type;
use arrow::array::{
    new_null_array, Array, ArrayBuilder, ArrayRef, BinaryBuilder, BooleanBuilder, Date32Builder,
    Decimal128Builder, FixedSizeListBuilder, Float32Builder, Float64Builder, Int16Builder,
    Int32Builder, Int64Builder, Int8Builder, IntervalMonthDayNanoBuilder, LargeBinaryBuilder,
    LargeStringBuilder, ListBuilder, RecordBatch, RecordBatchOptions, StringArray, StringBuilder,
    StringDictionaryBuilder, StructBuilder, Time64NanosecondBuilder, TimestampNanosecondBuilder,
    UInt32Builder,
};
use arrow::datatypes::{
    DataType, Date32Type, Field, Int8Type, IntervalMonthDayNanoType, IntervalUnit, Schema,
    SchemaRef, TimeUnit,
};
use arrow_json::ReaderBuilder;
use bigdecimal::BigDecimal;
use byteorder::{BigEndian, ReadBytesExt};
use chrono::{DateTime, Timelike, Utc};
use composite::CompositeType;
use geo_types::geometry::Point;
use rust_decimal::Decimal;
use sea_query::{Alias, ColumnType, SeaRc};
use snafu::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_postgres::types::FromSql;
use tokio_postgres::types::Kind;
use tokio_postgres::{types::Type, Row};

pub mod builder;
pub mod composite;
pub mod hive_schema;
pub mod schema;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to build record batch: {source}"))]
    FailedToBuildRecordBatch {
        source: datafusion::arrow::error::ArrowError,
    },

    #[snafu(display("No builder found for index {index}"))]
    NoBuilderForIndex { index: usize },

    #[snafu(display(
        "Failed to read column '{column}': a value needs more than the {column_precision} digits the column carries once it is given {column_scale} decimal places. \
        Declare the column with a smaller scale at the source, or read it as text."
    ))]
    NumericValueTooLarge {
        column: String,
        column_precision: u8,
        column_scale: i8,
    },

    #[snafu(display(
        "Failed to read column '{column}': the value {value} cannot be represented as {target}, the type the query expects for this column. \
        Cast the expression to a wider type at the source, or declare the column with a type that holds the value."
    ))]
    NumericNotRepresentable {
        column: String,
        value: String,
        target: DataType,
    },

    #[snafu(display("Failed to downcast builder for {postgres_type}"))]
    FailedToDowncastBuilder { postgres_type: String },

    #[snafu(display("Integer overflow when converting u64 to i64: {source}"))]
    FailedToConvertU64toI64 {
        source: <u64 as convert::TryInto<i64>>::Error,
    },

    #[snafu(display("Integer overflow when converting u128 to i64: {source}"))]
    FailedToConvertU128toI64 {
        source: <u128 as convert::TryInto<i64>>::Error,
    },

    #[snafu(display("Failed to get a row value for {pg_type}: {source}"))]
    FailedToGetRowValue {
        pg_type: Type,
        source: tokio_postgres::Error,
    },

    #[snafu(display("Failed to get a composite row value for {pg_type}: {source}"))]
    FailedToGetCompositeRowValue {
        pg_type: Type,
        source: composite::Error,
    },

    #[snafu(display("Failed to parse raw Postgres Bytes as BigDecimal: {:?}", bytes))]
    FailedToParseBigDecimalFromPostgres { bytes: Vec<u8> },

    #[snafu(display("Cannot represent BigDecimal as i128: {big_decimal}"))]
    FailedToConvertBigDecimalToI128 { big_decimal: BigDecimal },

    #[snafu(display("Failed to find field {column_name} in schema"))]
    FailedToFindFieldInSchema { column_name: String },

    #[snafu(display("No Arrow field found for index {index}"))]
    NoArrowFieldForIndex { index: usize },

    #[snafu(display("No PostgreSQL scale found for index {index}"))]
    NoPostgresScaleForIndex { index: usize },

    #[snafu(display("No column name for index: {index}"))]
    NoColumnNameForIndex { index: usize },

    #[snafu(display(
        "Expected Utf8 intermediate array for JSON List<Struct> column '{column_name}'"
    ))]
    InvalidJsonListStructIntermediateArray { column_name: String },

    #[snafu(display("Failed to decode JSON List<Struct> for column '{column_name}': {source}"))]
    FailedToDecodeJsonListStruct {
        column_name: String,
        source: arrow::error::ArrowError,
    },

    #[snafu(display("The field '{field_name}' has an unsupported data type: {data_type}."))]
    UnsupportedDataType {
        data_type: String,
        field_name: String,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

macro_rules! handle_primitive_type {
    ($builder:expr, $type:expr, $builder_ty:ty, $value_ty:ty, $row:expr, $index:expr) => {{
        let Some(builder) = $builder else {
            return NoBuilderForIndexSnafu { index: $index }.fail();
        };
        let Some(builder) = builder.as_any_mut().downcast_mut::<$builder_ty>() else {
            return FailedToDowncastBuilderSnafu {
                postgres_type: format!("{:?}", $type),
            }
            .fail();
        };
        let v: Option<$value_ty> = $row
            .try_get($index)
            .context(FailedToGetRowValueSnafu { pg_type: $type })?;

        match v {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }};
}

macro_rules! handle_primitive_array_type {
    ($type:expr, $builder:expr, $row:expr, $i:expr, $list_builder:ty, $value_type:ty) => {{
        let Some(builder) = $builder else {
            return NoBuilderForIndexSnafu { index: $i }.fail();
        };
        let Some(builder) = builder.as_any_mut().downcast_mut::<$list_builder>() else {
            return FailedToDowncastBuilderSnafu {
                postgres_type: format!("{:?}", $type),
            }
            .fail();
        };
        let v: Option<Vec<$value_type>> = $row
            .try_get($i)
            .context(FailedToGetRowValueSnafu { pg_type: $type })?;
        match v {
            Some(v) => {
                let v = v.into_iter().map(Some);
                builder.append_value(v);
            }
            None => builder.append_null(),
        }
    }};
}

macro_rules! handle_composite_type {
    ($BuilderType:ty, $ValueType:ty, $pg_type:expr, $composite_type:expr, $builder:expr, $idx:expr, $field_name:expr) => {{
        let Some(field_builder) = $builder.field_builder::<$BuilderType>($idx) else {
            return FailedToDowncastBuilderSnafu {
                postgres_type: format!("{}", $pg_type),
            }
            .fail();
        };
        let v: Option<$ValueType> =
            $composite_type
                .try_get($field_name)
                .context(FailedToGetCompositeRowValueSnafu {
                    pg_type: $pg_type.clone(),
                })?;
        match v {
            Some(v) => field_builder.append_value(v),
            None => field_builder.append_null(),
        }
    }};
}

macro_rules! handle_composite_types {
    ($field_type:expr, $pg_type:expr, $composite_type:expr, $builder:expr, $idx:expr, $field_name:expr, $($DataType:ident => ($BuilderType:ty, $ValueType:ty)),*) => {
        match $field_type {
            $(
                DataType::$DataType => {
                    handle_composite_type!(
                        $BuilderType,
                        $ValueType,
                        $pg_type,
                        $composite_type,
                        $builder,
                        $idx,
                        $field_name
                    );
                }
            )*
            _ => unimplemented!("Unsupported field type {:?}", $field_type),
        }
    }
}

/// Appends every field of a `CompositeType` value into `$struct_builder`'s field builders
/// (a single struct row). The caller is responsible for the matching
/// `$struct_builder.append(...)` validity call. Shared by the top-level composite column
/// path and the composite-array (`List<Struct>`) element path.
macro_rules! append_composite_fields_to_struct {
    ($composite_type:expr, $struct_builder:expr) => {{
        let fields = $composite_type.fields();
        for (idx, field) in fields.iter().enumerate() {
            let field_name = field.name();
            let Some(field_type) = map_column_type_to_data_type(field.type_(), field_name)? else {
                return UnsupportedDataTypeSnafu {
                    data_type: field.type_().to_string(),
                    field_name: field_name.to_string(),
                }
                .fail();
            };

            handle_composite_types!(
                field_type,
                field.type_(),
                $composite_type,
                $struct_builder,
                idx,
                field_name,
                Boolean => (BooleanBuilder, bool),
                Int8 => (Int8Builder, i8),
                Int16 => (Int16Builder, i16),
                Int32 => (Int32Builder, i32),
                Int64 => (Int64Builder, i64),
                UInt32 => (UInt32Builder, u32),
                Float32 => (Float32Builder, f32),
                Float64 => (Float64Builder, f64),
                Binary => (BinaryBuilder, Vec<u8>),
                LargeBinary => (LargeBinaryBuilder, Vec<u8>),
                Utf8 => (StringBuilder, String),
                LargeUtf8 => (LargeStringBuilder, String)
            );
        }
    }};
}

/// Arrow type for a PostgreSQL `NUMERIC` whose precision and scale the schema
/// does not declare.
///
/// An unconstrained `NUMERIC` has no column-level precision or scale at all —
/// every value carries its own — while an Arrow `Decimal128` column has exactly
/// one of each, so a single pair has to stand for the whole column.
///
/// The catalog settles on this same pair (`pg_data_type_to_arrow_type`), and
/// this is what a caller with no projected schema gets, so the two agree on the
/// column rather than describing it differently.
///
/// The pair must not be read off the data. The schema a caller has here may
/// come from sampling a single row (`infer_schema_from_data` runs
/// `SELECT * FROM <table> LIMIT 1` when the catalog reports no columns), and a
/// scale taken from one row pins the column to whatever that row happened to
/// hold, silently rescaling every other value: `1.23456` beside a `1.5` comes
/// back as `1.2`, and because a NULL carries no scale, a column sampled on a
/// NULL row comes back with every fraction truncated. That sample is not stable
/// either — `LIMIT 1` has no `ORDER BY`.
///
/// A fixed pair keeps the column deterministic and carries every value with up
/// to 20 decimal places exactly; a value needing more is rounded to it (see
/// `numeric_text_coefficient`).
const NUMERIC_UNDECLARED_PRECISION: u8 = 38;
const NUMERIC_UNDECLARED_SCALE: i8 = 20;

/// Why a `NUMERIC` value cannot be carried by the column it was read into.
#[derive(Debug, PartialEq, Eq)]
enum NumericFit {
    /// The value needs more digits than the column's precision even after
    /// rounding to its scale.
    PrecisionTooNarrow,
}

/// Converts Postgres `Row`s to an Arrow `RecordBatch`. Assumes that all rows have the same schema and
/// sets the schema based on the first row.
///
/// # Errors
///
/// Returns an error if there is a failure in converting the rows to a `RecordBatch`.
/// Nanoseconds since the Unix epoch, negative for a time before it.
///
/// `SystemTime::duration_since` fails for a time before the epoch, so that case is the
/// distance back to it. Skipping it instead leaves the column one value short of the
/// others, and building the record batch then fails for every row read with it.
fn unix_nanos(time: SystemTime) -> Result<i64> {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_nanos()),
        Err(before) => i64::try_from(before.duration().as_nanos()).map(|nanos| -nanos),
    }
    .context(FailedToConvertU128toI64Snafu)
}

#[allow(clippy::too_many_lines)]
pub fn rows_to_arrow(rows: &[Row], projected_schema: &Option<SchemaRef>) -> Result<RecordBatch> {
    let mut arrow_fields: Vec<Option<Field>> = Vec::new();
    let mut arrow_columns_builders: Vec<Option<Box<dyn ArrayBuilder>>> = Vec::new();
    let mut postgres_types: Vec<Type> = Vec::new();
    let mut postgres_numeric_scales: Vec<Option<u32>> = Vec::new();
    let mut column_names: Vec<String> = Vec::new();
    let mut projected_json_complex_fields: Vec<Option<Arc<Field>>> = Vec::new();

    if !rows.is_empty() {
        let row = &rows[0];
        let column_count = row.columns().len();
        for (column_index, column) in row.columns().iter().enumerate() {
            let column_name = column.name();
            let column_type = column.type_();
            let projected_field = projected_destination_field(
                projected_schema.as_ref(),
                column_name,
                column_index,
                column_count,
            );
            let projected_json_complex_field =
                projected_json_complex_field(projected_field, column_type);

            let mut numeric_scale: Option<u32> = None;

            let mut data_type = if *column_type == Type::NUMERIC {
                let destination = projected_destination_field(
                    projected_schema.as_ref(),
                    column_name,
                    column_index,
                    column_count,
                )
                .map(Field::data_type);
                match destination {
                    Some(DataType::Decimal128(precision, scale)) => {
                        numeric_scale = Some(u32::try_from(*scale).unwrap_or_default());
                        Some(DataType::Decimal128(*precision, *scale))
                    }
                    // The plan has already committed to a type that is not a
                    // decimal, so produce it directly from the value's own
                    // digits rather than through a `Decimal128` whose fixed
                    // precision the value may not fit. See `NumericText` and
                    // `append_numeric_to_destination`, which reads every
                    // scalar `NUMERIC` from those digits.
                    Some(
                        destination @ (DataType::Float64 | DataType::Float32 | DataType::Int64),
                    ) => Some(destination.clone()),
                    // Undeclared scale: see `NUMERIC_UNDECLARED_SCALE`.
                    _ => {
                        numeric_scale =
                            Some(u32::try_from(NUMERIC_UNDECLARED_SCALE).unwrap_or_default());
                        Some(DataType::Decimal128(
                            NUMERIC_UNDECLARED_PRECISION,
                            NUMERIC_UNDECLARED_SCALE,
                        ))
                    }
                }
            } else if *column_type == Type::NUMERIC_ARRAY {
                if let Some(schema) = projected_schema.as_ref() {
                    match get_decimal_array_column_precision_and_scale(column_name, schema) {
                        Some((precision, scale)) => {
                            numeric_scale = Some(u32::try_from(scale).unwrap_or_default());
                            Some(DataType::List(Arc::new(Field::new(
                                "item",
                                DataType::Decimal128(precision, scale),
                                true,
                            ))))
                        }
                        None => None,
                    }
                } else {
                    None
                }
            } else {
                map_column_type_to_data_type(column_type, column_name)?
            };

            let nullable = projected_field
                .map(|field| field.is_nullable())
                .unwrap_or(true);

            if projected_json_complex_field.is_some() {
                // Collect the JSON text in a temporary Utf8 builder and decode it into
                // the projected complex Arrow type after row collection.
                data_type = Some(DataType::Utf8);
            }

            match &data_type {
                Some(data_type) => {
                    arrow_fields.push(Some(Field::new(column_name, data_type.clone(), nullable)));
                }
                None => arrow_fields.push(None),
            }
            postgres_numeric_scales.push(numeric_scale);
            arrow_columns_builders
                .push(map_data_type_to_array_builder_optional(data_type.as_ref()));
            postgres_types.push(column_type.clone());
            column_names.push(column_name.to_string());
            projected_json_complex_fields.push(projected_json_complex_field);
        }
    }

    for row in rows {
        for (i, postgres_type) in postgres_types.iter().enumerate() {
            let Some(builder) = arrow_columns_builders.get_mut(i) else {
                return NoBuilderForIndexSnafu { index: i }.fail();
            };

            let Some(arrow_field) = arrow_fields.get_mut(i) else {
                return NoArrowFieldForIndexSnafu { index: i }.fail();
            };

            let Some(postgres_numeric_scale) = postgres_numeric_scales.get_mut(i) else {
                return NoPostgresScaleForIndexSnafu { index: i }.fail();
            };

            match *postgres_type {
                Type::INT2 => {
                    handle_primitive_type!(builder, Type::INT2, Int16Builder, i16, row, i);
                }
                Type::INT4 => {
                    handle_primitive_type!(builder, Type::INT4, Int32Builder, i32, row, i);
                }
                Type::INT8 => {
                    handle_primitive_type!(builder, Type::INT8, Int64Builder, i64, row, i);
                }
                Type::OID => {
                    handle_primitive_type!(builder, Type::OID, UInt32Builder, u32, row, i);
                }
                Type::XID => {
                    handle_primitive_type!(builder, Type::XID, UInt32Builder, u32, row, i);
                }
                Type::FLOAT4 => {
                    handle_primitive_type!(builder, Type::FLOAT4, Float32Builder, f32, row, i);
                }
                Type::FLOAT8 => {
                    handle_primitive_type!(builder, Type::FLOAT8, Float64Builder, f64, row, i);
                }
                Type::CHAR => {
                    handle_primitive_type!(builder, Type::CHAR, Int8Builder, i8, row, i);
                }
                Type::TEXT => {
                    handle_primitive_type!(builder, Type::TEXT, StringBuilder, &str, row, i);
                }
                Type::VARCHAR => {
                    handle_primitive_type!(builder, Type::VARCHAR, StringBuilder, &str, row, i);
                }
                Type::NAME => {
                    handle_primitive_type!(builder, Type::NAME, StringBuilder, &str, row, i);
                }
                Type::BYTEA => {
                    handle_primitive_type!(builder, Type::BYTEA, BinaryBuilder, Vec<u8>, row, i);
                }
                Type::BPCHAR => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v: Option<&str> = row.try_get(i).context(FailedToGetRowValueSnafu {
                        pg_type: Type::BPCHAR,
                    })?;

                    match v {
                        Some(v) => builder.append_value(v.trim_end()),
                        None => builder.append_null(),
                    }
                }
                Type::BOOL => {
                    handle_primitive_type!(builder, Type::BOOL, BooleanBuilder, bool, row, i);
                }
                Type::MONEY => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row
                        .try_get::<usize, Option<MoneyFromSql>>(i)
                        .with_context(|_| FailedToGetRowValueSnafu {
                            pg_type: Type::MONEY,
                        })?;

                    match v {
                        Some(v) => {
                            builder.append_value(v.cash_value);
                        }
                        None => builder.append_null(),
                    }
                }
                // Schema validation will only allow JSONB columns when `UnsupportedTypeAction` is set to `String`, so it is safe to handle JSONB here as strings.
                Type::JSON | Type::JSONB => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row
                        .try_get::<usize, Option<JsonbRawString>>(i)
                        .with_context(|_| FailedToGetRowValueSnafu {
                            pg_type: postgres_type.clone(),
                        })?;

                    match v {
                        Some(v) => builder.append_value(v.0),
                        None => builder.append_null(),
                    }
                }
                Type::TIME => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder
                        .as_any_mut()
                        .downcast_mut::<Time64NanosecondBuilder>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row
                        .try_get::<usize, Option<chrono::NaiveTime>>(i)
                        .with_context(|_| FailedToGetRowValueSnafu {
                            pg_type: Type::TIME,
                        })?;

                    match v {
                        Some(v) => {
                            let timestamp: i64 = i64::from(v.num_seconds_from_midnight())
                                * 1_000_000_000
                                + i64::from(v.nanosecond());
                            builder.append_value(timestamp);
                        }
                        None => builder.append_null(),
                    }
                }
                Type::POINT => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder
                        .as_any_mut()
                        .downcast_mut::<FixedSizeListBuilder<Float64Builder>>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };

                    let v = row.try_get::<usize, Option<Point>>(i).with_context(|_| {
                        FailedToGetRowValueSnafu {
                            pg_type: Type::POINT,
                        }
                    })?;

                    if let Some(v) = v {
                        builder.values().append_value(v.x());
                        builder.values().append_value(v.y());
                        builder.append(true);
                    } else {
                        builder.values().append_null();
                        builder.values().append_null();
                        builder.append(false);
                    }
                }
                Type::INTERVAL => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder
                        .as_any_mut()
                        .downcast_mut::<IntervalMonthDayNanoBuilder>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };

                    let v: Option<IntervalFromSql> =
                        row.try_get(i).context(FailedToGetRowValueSnafu {
                            pg_type: Type::INTERVAL,
                        })?;
                    match v {
                        Some(v) => {
                            let interval_month_day_nano = IntervalMonthDayNanoType::make_value(
                                v.month,
                                v.day,
                                v.time * 1_000,
                            );
                            builder.append_value(interval_month_day_nano);
                        }
                        None => builder.append_null(),
                    }
                }
                Type::NUMERIC => {
                    let v: Option<NumericText> =
                        row.try_get(i).context(FailedToGetRowValueSnafu {
                            pg_type: Type::NUMERIC,
                        })?;
                    let Some(field) = arrow_field.as_ref() else {
                        return NoArrowFieldForIndexSnafu { index: i }.fail();
                    };
                    append_numeric_to_destination(builder, i, field, v)?;
                }
                Type::NUMERIC_ARRAY => {
                    let v: Option<Vec<Option<Decimal>>> =
                        row.try_get(i).context(FailedToGetRowValueSnafu {
                            pg_type: Type::NUMERIC_ARRAY,
                        })?;

                    let inferred_scale = v
                        .iter()
                        .flatten()
                        .flatten()
                        .map(Decimal::scale)
                        .max()
                        .unwrap_or_default();

                    let dest_scale = postgres_numeric_scale.unwrap_or(inferred_scale);
                    let decimal_scale = i8::try_from(dest_scale).unwrap_or_default();

                    let decimal_array_builder = builder.get_or_insert_with(|| {
                        Box::new(ListBuilder::new(
                            Decimal128Builder::new()
                                .with_precision_and_scale(38, decimal_scale)
                                .unwrap_or_default(),
                        ))
                    });

                    let Some(decimal_array_builder) = decimal_array_builder
                        .as_any_mut()
                        .downcast_mut::<ListBuilder<Decimal128Builder>>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };

                    if arrow_field.is_none() {
                        let Some(field_name) = column_names.get(i) else {
                            return NoColumnNameForIndexSnafu { index: i }.fail();
                        };

                        let new_arrow_field = Field::new(
                            field_name,
                            DataType::List(Arc::new(Field::new(
                                "item",
                                DataType::Decimal128(38, decimal_scale),
                                true,
                            ))),
                            true,
                        );

                        *arrow_field = Some(new_arrow_field);
                    }

                    if postgres_numeric_scale.is_none() {
                        *postgres_numeric_scale = Some(dest_scale);
                    };

                    let Some(values) = v else {
                        decimal_array_builder.append_null();
                        continue;
                    };

                    for item in values {
                        if let Some(mut decimal) = item {
                            decimal.rescale(dest_scale);
                            decimal_array_builder
                                .values()
                                .append_value(decimal.mantissa());
                        } else {
                            decimal_array_builder.values().append_null();
                        }
                    }
                    decimal_array_builder.append(true);
                }
                Type::TIMESTAMP => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder
                        .as_any_mut()
                        .downcast_mut::<TimestampNanosecondBuilder>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row
                        .try_get::<usize, Option<SystemTime>>(i)
                        .with_context(|_| FailedToGetRowValueSnafu {
                            pg_type: Type::TIMESTAMP,
                        })?;

                    match v {
                        Some(v) => builder.append_value(unix_nanos(v)?),
                        None => builder.append_null(),
                    }
                }
                Type::TIMESTAMPTZ => {
                    let v = row
                        .try_get::<usize, Option<DateTime<Utc>>>(i)
                        .with_context(|_| FailedToGetRowValueSnafu {
                            pg_type: Type::TIMESTAMPTZ,
                        })?;

                    let timestamptz_builder = builder.get_or_insert_with(|| {
                        Box::new(TimestampNanosecondBuilder::new().with_timezone("UTC"))
                    });

                    let Some(timestamptz_builder) = timestamptz_builder
                        .as_any_mut()
                        .downcast_mut::<TimestampNanosecondBuilder>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };

                    if arrow_field.is_none() {
                        let Some(field_name) = column_names.get(i) else {
                            return NoColumnNameForIndexSnafu { index: i }.fail();
                        };
                        let new_arrow_field = Field::new(
                            field_name,
                            DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
                            true,
                        );

                        *arrow_field = Some(new_arrow_field);
                    }

                    match v {
                        Some(v) => {
                            let utc_timestamp =
                                v.to_utc().timestamp_nanos_opt().unwrap_or_default();
                            timestamptz_builder.append_value(utc_timestamp);
                        }
                        None => timestamptz_builder.append_null(),
                    }
                }

                Type::DATE => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<Date32Builder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row.try_get::<usize, Option<chrono::NaiveDate>>(i).context(
                        FailedToGetRowValueSnafu {
                            pg_type: Type::DATE,
                        },
                    )?;

                    match v {
                        Some(v) => builder.append_value(Date32Type::from_naive_date(v)),
                        None => builder.append_null(),
                    }
                }
                Type::UUID => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row.try_get::<usize, Option<uuid::Uuid>>(i).context(
                        FailedToGetRowValueSnafu {
                            pg_type: Type::UUID,
                        },
                    )?;

                    match v {
                        Some(v) => builder.append_value(v.to_string()),
                        None => builder.append_null(),
                    }
                }
                Type::INT2_ARRAY => handle_primitive_array_type!(
                    Type::INT2_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<Int16Builder>,
                    i16
                ),
                Type::INT4_ARRAY => handle_primitive_array_type!(
                    Type::INT4_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<Int32Builder>,
                    i32
                ),
                Type::INT8_ARRAY => handle_primitive_array_type!(
                    Type::INT8_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<Int64Builder>,
                    i64
                ),
                Type::OID_ARRAY => handle_primitive_array_type!(
                    Type::OID_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<UInt32Builder>,
                    u32
                ),
                Type::FLOAT4_ARRAY => handle_primitive_array_type!(
                    Type::FLOAT4_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<Float32Builder>,
                    f32
                ),
                Type::FLOAT8_ARRAY => handle_primitive_array_type!(
                    Type::FLOAT8_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<Float64Builder>,
                    f64
                ),
                Type::TEXT_ARRAY => handle_primitive_array_type!(
                    Type::TEXT_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<StringBuilder>,
                    String
                ),
                Type::BOOL_ARRAY => handle_primitive_array_type!(
                    Type::BOOL_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<BooleanBuilder>,
                    bool
                ),
                Type::BYTEA_ARRAY => handle_primitive_array_type!(
                    Type::BYTEA_ARRAY,
                    builder,
                    row,
                    i,
                    ListBuilder<BinaryBuilder>,
                    Vec<u8>
                ),
                _ if matches!(postgres_type.name(), "geometry" | "geography") => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<BinaryBuilder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row.try_get::<usize, Option<GeometryFromSql>>(i).context(
                        FailedToGetRowValueSnafu {
                            pg_type: postgres_type.clone(),
                        },
                    )?;

                    match v {
                        Some(v) => builder.append_value(v.wkb),
                        None => builder.append_null(),
                    }
                }
                _ if matches!(postgres_type.name(), "_geometry" | "_geography") => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder
                        .as_any_mut()
                        .downcast_mut::<ListBuilder<BinaryBuilder>>()
                    else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v: Option<Vec<GeometryFromSql>> =
                        row.try_get(i).context(FailedToGetRowValueSnafu {
                            pg_type: postgres_type.clone(),
                        })?;
                    match v {
                        Some(v) => {
                            let v = v.into_iter().map(|item| Some(item.wkb));
                            builder.append_value(v);
                        }
                        None => builder.append_null(),
                    }
                }
                // Redshift `SUPER` (and Spectrum `ARRAY`/`STRUCT`/`MAP` external columns
                // surfaced as `SUPER`) arrive as JSON text. Collect that text into a
                // `Utf8` builder; if the column is projected as a complex Arrow type it is
                // decoded into that type after row collection (see
                // `projected_json_complex_field`), otherwise it stays a JSON string.
                _ if postgres_type.name() == "super" => {
                    let Some(builder) = builder else {
                        return NoBuilderForIndexSnafu { index: i }.fail();
                    };
                    let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() else {
                        return FailedToDowncastBuilderSnafu {
                            postgres_type: format!("{postgres_type}"),
                        }
                        .fail();
                    };
                    let v = row.try_get::<usize, Option<SuperRawString>>(i).context(
                        FailedToGetRowValueSnafu {
                            pg_type: postgres_type.clone(),
                        },
                    )?;

                    match v {
                        Some(v) => builder.append_value(v.0),
                        None => builder.append_null(),
                    }
                }
                _ => match *postgres_type.kind() {
                    // Array of a composite type (`my_struct[]`) → List<Struct>. Each array
                    // element is a `CompositeType`; append it as a struct row in the list.
                    Kind::Array(ref element_type)
                        if matches!(*element_type.kind(), Kind::Composite(_)) =>
                    {
                        let Some(builder) = builder else {
                            return NoBuilderForIndexSnafu { index: i }.fail();
                        };
                        let Some(list_builder) = builder
                            .as_any_mut()
                            .downcast_mut::<ListBuilder<StructBuilder>>()
                        else {
                            return FailedToDowncastBuilderSnafu {
                                postgres_type: format!("{postgres_type}"),
                            }
                            .fail();
                        };

                        let v = row
                            .try_get::<usize, Option<Vec<CompositeType>>>(i)
                            .context(FailedToGetRowValueSnafu {
                                pg_type: postgres_type.clone(),
                            })?;

                        let Some(composites) = v else {
                            list_builder.append_null();
                            continue;
                        };

                        let struct_builder = list_builder.values();
                        for composite_type in &composites {
                            append_composite_fields_to_struct!(composite_type, struct_builder);
                            struct_builder.append(true);
                        }
                        list_builder.append(true);
                    }
                    Kind::Composite(_) => {
                        let Some(builder) = builder else {
                            return NoBuilderForIndexSnafu { index: i }.fail();
                        };
                        let Some(builder) = builder.as_any_mut().downcast_mut::<StructBuilder>()
                        else {
                            return FailedToDowncastBuilderSnafu {
                                postgres_type: format!("{postgres_type}"),
                            }
                            .fail();
                        };

                        let v = row.try_get::<usize, Option<CompositeType>>(i).context(
                            FailedToGetRowValueSnafu {
                                pg_type: postgres_type.clone(),
                            },
                        )?;

                        let Some(composite_type) = v else {
                            builder.append_null();
                            continue;
                        };

                        builder.append(true);

                        append_composite_fields_to_struct!(composite_type, builder);
                    }
                    Kind::Enum(_) => {
                        let Some(builder) = builder else {
                            return NoBuilderForIndexSnafu { index: i }.fail();
                        };
                        let Some(builder) = builder
                            .as_any_mut()
                            .downcast_mut::<StringDictionaryBuilder<Int8Type>>()
                        else {
                            return FailedToDowncastBuilderSnafu {
                                postgres_type: format!("{postgres_type}"),
                            }
                            .fail();
                        };

                        let v = row.try_get::<usize, Option<EnumValueFromSql>>(i).context(
                            FailedToGetRowValueSnafu {
                                pg_type: postgres_type.clone(),
                            },
                        )?;

                        match v {
                            Some(v) => builder.append_value(v.enum_value),
                            None => builder.append_null(),
                        }
                    }
                    _ => {
                        return UnsupportedDataTypeSnafu {
                            data_type: postgres_type.to_string(),
                            field_name: column_names[i].clone(),
                        }
                        .fail();
                    }
                },
            }
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::new();
    let mut finalized_fields: Vec<Field> = Vec::new();
    for (i, builder) in arrow_columns_builders.into_iter().enumerate() {
        let Some(mut builder) = builder else {
            continue;
        };

        let mut array = builder.finish();
        let Some(mut arrow_field) = arrow_fields.get(i).cloned().flatten() else {
            return NoArrowFieldForIndexSnafu { index: i }.fail();
        };

        if let Some(projected_field) = projected_json_complex_fields.get(i).cloned().flatten() {
            let Some(string_array) = array.as_any().downcast_ref::<StringArray>() else {
                return InvalidJsonListStructIntermediateArraySnafu {
                    column_name: projected_field.name().to_string(),
                }
                .fail();
            };

            array = decode_json_complex_column(string_array, projected_field.as_ref()).context(
                FailedToDecodeJsonListStructSnafu {
                    column_name: projected_field.name().to_string(),
                },
            )?;
            arrow_field = projected_field.as_ref().clone();
        }

        columns.push(array);
        finalized_fields.push(arrow_field);
    }

    let options = &RecordBatchOptions::new().with_row_count(Some(rows.len()));
    match RecordBatch::try_new_with_options(
        Arc::new(Schema::new(finalized_fields)),
        columns,
        options,
    ) {
        Ok(record_batch) => Ok(record_batch),
        Err(e) => Err(e).context(FailedToBuildRecordBatchSnafu),
    }
}

/// Identifies columns whose values arrive as a JSON text serialization of a complex
/// Arrow type and should be decoded post-collection rather than read as a scalar.
///
/// Two sources produce these:
/// - PostgreSQL `JSON`/`JSONB` columns projected as a complex Arrow type.
/// - Redshift Spectrum external `ARRAY`/`STRUCT`/`MAP` columns, which Redshift serializes
///   to `VARCHAR(65535)` JSON text over the wire (see `json_serialization_enable`).
///
/// Returns the projected field when the wire column is text-like *and* the projected
/// Arrow type is a complex type (`List`/`LargeList`/`Struct`/`Map`). Such a pairing only
/// occurs for these JSON-text cases — ordinary text columns project as `Utf8`, and native
/// composite/array columns carry their own wire OIDs handled elsewhere — so this is safe
/// to apply regardless of the source database variant.
fn projected_json_complex_field(
    projected_field: Option<&Field>,
    column_type: &Type,
) -> Option<Arc<Field>> {
    // Only text-bearing wire types can carry a JSON serialization. These all have a
    // string-producing row arm above, so forcing the collection builder to `Utf8` is safe.
    // Redshift's `super` (matched by name — it has no stable built-in OID) is how Spectrum
    // surfaces serialized `ARRAY`/`STRUCT`/`MAP` external columns.
    let is_text_like = matches!(
        *column_type,
        Type::JSON | Type::JSONB | Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME
    ) || column_type.name() == "super";
    if !is_text_like {
        return None;
    }

    let field = Arc::new(projected_field?.clone());
    match field.data_type() {
        DataType::List(_) | DataType::LargeList(_) | DataType::Struct(_) | DataType::Map(_, _) => {
            Some(field)
        }
        _ => None,
    }
}

/// Decodes a `StringArray` of JSON values into a typed complex Arrow array (`List`,
/// `Struct`, `Map`, …) using a single batch NDJSON decode against `field`'s data type.
///
/// `arrow_json`'s field decoder treats each JSON line as a value of `field.data_type()`
/// (arrays decode `[...]`, structs/maps decode `{...}`), producing a single-column batch
/// whose `column(0)` is the decoded array. Null entries in `string_array` are emitted as
/// the JSON literal `null`, which the decoder interprets as a null element — no post-hoc
/// `take` reindexing required.
fn decode_json_complex_column(
    string_array: &StringArray,
    field: &Field,
) -> std::result::Result<ArrayRef, arrow::error::ArrowError> {
    // The field name is unused for decoding — the caller overwrites the field from the
    // projected schema.  We only need the data type for the decoder.
    let decode_field = Arc::new(field.clone());

    if string_array.is_empty() {
        return Ok(new_null_array(decode_field.data_type(), 0));
    }

    if string_array.null_count() == string_array.len() {
        return Ok(new_null_array(decode_field.data_type(), string_array.len()));
    }

    let mut decoder = ReaderBuilder::new_with_field(decode_field)
        .with_batch_size(string_array.len())
        .build_decoder()
        .map_err(|e| {
            arrow::error::ArrowError::CastError(format!("Failed to create decoder: {e}"))
        })?;

    // Build NDJSON buffer: non-null rows get their JSON, null rows get "null".
    let ndjson_capacity: usize = string_array
        .iter()
        .map(|value| value.map_or(4, str::len) + 1)
        .sum();
    let mut ndjson = Vec::with_capacity(ndjson_capacity);
    for value in string_array {
        match value {
            Some(s) => ndjson.extend_from_slice(s.as_bytes()),
            None => ndjson.extend_from_slice(b"null"),
        }
        ndjson.push(b'\n');
    }

    decoder
        .decode(&ndjson)
        .map_err(|e| arrow::error::ArrowError::CastError(format!("Failed to decode value: {e}")))?;

    let batch = decoder.flush().map_err(|e| {
        arrow::error::ArrowError::CastError(format!("Failed to flush JSON decoder: {e}"))
    })?;

    match batch {
        Some(batch) if batch.num_rows() == string_array.len() => Ok(Arc::clone(batch.column(0))),
        Some(batch) => Err(arrow::error::ArrowError::CastError(format!(
            "expected {} rows, got {}",
            string_array.len(),
            batch.num_rows()
        ))),
        None => Err(arrow::error::ArrowError::CastError(
            "JSON decoder produced no output for non-empty input".into(),
        )),
    }
}

fn map_column_type_to_data_type(column_type: &Type, field_name: &str) -> Result<Option<DataType>> {
    match *column_type {
        Type::INT2 => Ok(Some(DataType::Int16)),
        Type::INT4 => Ok(Some(DataType::Int32)),
        Type::INT8 | Type::MONEY => Ok(Some(DataType::Int64)),
        Type::OID | Type::XID => Ok(Some(DataType::UInt32)),
        Type::FLOAT4 => Ok(Some(DataType::Float32)),
        Type::FLOAT8 => Ok(Some(DataType::Float64)),
        Type::CHAR => Ok(Some(DataType::Int8)),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::UUID | Type::NAME => {
            Ok(Some(DataType::Utf8))
        }
        Type::BYTEA => Ok(Some(DataType::Binary)),
        Type::BOOL => Ok(Some(DataType::Boolean)),
        // Schema validation will only allow JSONB columns when `UnsupportedTypeAction` is set to `String`, so it is safe to handle JSONB here as strings.
        Type::JSON | Type::JSONB => Ok(Some(DataType::Utf8)),
        // Inspect the scale from the first row. Precision will always be 38 for Decimal128.
        Type::NUMERIC => Ok(None),
        // Inspect the scale from the first row. Precision will always be 38 for Decimal128.
        Type::NUMERIC_ARRAY => Ok(None),
        Type::TIMESTAMPTZ => Ok(Some(DataType::Timestamp(
            TimeUnit::Nanosecond,
            Some(Arc::from("UTC")),
        ))),
        // We get a SystemTime that we can always convert into milliseconds
        Type::TIMESTAMP => Ok(Some(DataType::Timestamp(TimeUnit::Nanosecond, None))),
        Type::DATE => Ok(Some(DataType::Date32)),
        Type::TIME => Ok(Some(DataType::Time64(TimeUnit::Nanosecond))),
        Type::INTERVAL => Ok(Some(DataType::Interval(IntervalUnit::MonthDayNano))),
        Type::POINT => Ok(Some(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, true)),
            2,
        ))),
        Type::PG_NODE_TREE => Ok(Some(DataType::Utf8)),
        Type::INT2_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int16,
            true,
        ))))),
        Type::INT4_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            true,
        ))))),
        Type::INT8_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int64,
            true,
        ))))),
        Type::OID_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::UInt32,
            true,
        ))))),
        Type::FLOAT4_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Float32,
            true,
        ))))),
        Type::FLOAT8_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Float64,
            true,
        ))))),
        Type::TEXT_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        ))))),
        Type::BOOL_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Boolean,
            true,
        ))))),
        Type::BYTEA_ARRAY => Ok(Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Binary,
            true,
        ))))),
        _ if matches!(column_type.name(), "geometry" | "geography") => Ok(Some(DataType::Binary)),
        _ if matches!(column_type.name(), "_geometry" | "_geography") => Ok(Some(DataType::List(
            Arc::new(Field::new("item", DataType::Binary, true)),
        ))),
        // Redshift `SUPER` (and Spectrum complex external columns surfaced as `SUPER`)
        // arrive as JSON text. Default to `Utf8`; a complex projected schema upgrades it
        // to the decoded type via `projected_json_complex_field`.
        _ if column_type.name() == "super" => Ok(Some(DataType::Utf8)),
        _ => match *column_type.kind() {
            Kind::Composite(ref fields) => {
                let mut arrow_fields = Vec::new();
                for field in fields {
                    let field_name = field.name();
                    let field_type = map_column_type_to_data_type(field.type_(), field_name)?;
                    match field_type {
                        Some(field_type) => {
                            arrow_fields.push(Field::new(field_name, field_type, true));
                        }
                        None => {
                            return UnsupportedDataTypeSnafu {
                                data_type: field.type_().to_string(),
                                field_name: field_name.to_string(),
                            }
                            .fail();
                        }
                    }
                }
                Ok(Some(DataType::Struct(arrow_fields.into())))
            }
            Kind::Enum(_) => Ok(Some(DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Utf8),
            ))),
            // Array of a composite type (e.g. `my_struct[]`) → List<Struct>. The common
            // scalar arrays (`int[]`, `text[]`, …) are matched by their explicit
            // `Type::*_ARRAY` arms above; this catches user-defined composite arrays.
            Kind::Array(ref element_type) if matches!(*element_type.kind(), Kind::Composite(_)) => {
                let Some(element) = map_column_type_to_data_type(element_type, field_name)? else {
                    return UnsupportedDataTypeSnafu {
                        data_type: element_type.to_string(),
                        field_name: field_name.to_string(),
                    }
                    .fail();
                };
                Ok(Some(DataType::List(Arc::new(Field::new(
                    "item", element, true,
                )))))
            }
            _ => UnsupportedDataTypeSnafu {
                data_type: column_type.to_string(),
                field_name: field_name.to_string(),
            }
            .fail(),
        },
    }
}

pub(crate) fn map_data_type_to_column_type_postgres(
    data_type: &DataType,
    table_name: &str,
    field_name: &str,
) -> ColumnType {
    match data_type {
        DataType::Struct(_) => ColumnType::Custom(SeaRc::new(Alias::new(
            get_postgres_composite_type_name(table_name, field_name),
        ))),
        _ => map_data_type_to_column_type(data_type),
    }
}

#[must_use]
pub(crate) fn get_postgres_composite_type_name(table_name: &str, field_name: &str) -> String {
    format!("struct_{table_name}_{field_name}")
}

/// Extracts the raw JSON string from Postgres JSON/JSONB wire format without
/// parsing through `serde_json::Value`. JSONB prepends a `0x01` version byte
/// which is stripped; JSON is returned as-is.
#[derive(Debug)]
struct JsonbRawString(String);

impl<'a> FromSql<'a> for JsonbRawString {
    fn from_sql(
        ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let json_bytes = if *ty == Type::JSONB {
            if raw.is_empty() || raw[0] != 1 {
                return Err("unsupported JSONB encoding version".into());
            }
            &raw[1..]
        } else {
            raw
        };
        Ok(JsonbRawString(String::from_utf8(json_bytes.to_vec())?))
    }

    fn accepts(ty: &Type) -> bool {
        matches!(*ty, Type::JSON | Type::JSONB)
    }
}

/// Reads a Redshift `SUPER` value as its raw UTF-8 JSON text.
///
/// Redshift serializes `SUPER` — and the `ARRAY`/`STRUCT`/`MAP` columns of Spectrum
/// external tables, which surface over the wire as `SUPER` — to a JSON text
/// representation. `SUPER` has no stable built-in OID, so this matches by type name and
/// interprets the value bytes as UTF-8. (Requires `json_serialization_enable` on the
/// session for Spectrum complex columns to serialize rather than error server-side.)
#[derive(Debug)]
struct SuperRawString(String);

impl<'a> FromSql<'a> for SuperRawString {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(SuperRawString(String::from_utf8(raw.to_vec())?))
    }

    fn accepts(ty: &Type) -> bool {
        ty.name() == "super"
    }
}

// interval_send - Postgres C (https://github.com/postgres/postgres/blob/master/src/backend/utils/adt/timestamp.c#L1032)
// interval values are internally stored as three integral fields: months, days, and microseconds
struct IntervalFromSql {
    time: i64,
    day: i32,
    month: i32,
}

impl<'a> FromSql<'a> for IntervalFromSql {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let mut cursor = std::io::Cursor::new(raw);

        let time = cursor.read_i64::<BigEndian>()?;
        let day = cursor.read_i32::<BigEndian>()?;
        let month = cursor.read_i32::<BigEndian>()?;

        Ok(IntervalFromSql { time, day, month })
    }

    fn accepts(ty: &Type) -> bool {
        matches!(*ty, Type::INTERVAL)
    }
}

// cash_send - Postgres C (https://github.com/postgres/postgres/blob/bd8fe12ef3f727ed3658daf9b26beaf2b891e9bc/src/backend/utils/adt/cash.c#L603)
struct MoneyFromSql {
    cash_value: i64,
}

impl<'a> FromSql<'a> for MoneyFromSql {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let mut cursor = std::io::Cursor::new(raw);
        let cash_value = cursor.read_i64::<BigEndian>()?;
        Ok(MoneyFromSql { cash_value })
    }

    fn accepts(ty: &Type) -> bool {
        matches!(*ty, Type::MONEY)
    }
}

struct EnumValueFromSql {
    enum_value: String,
}

impl<'a> FromSql<'a> for EnumValueFromSql {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let mut cursor = std::io::Cursor::new(raw);
        let mut enum_value = String::new();
        cursor.read_to_string(&mut enum_value)?;
        Ok(EnumValueFromSql { enum_value })
    }

    fn accepts(ty: &Type) -> bool {
        matches!(*ty.kind(), Kind::Enum(_))
    }
}

pub struct GeometryFromSql<'a> {
    wkb: &'a [u8],
}

impl<'a> FromSql<'a> for GeometryFromSql<'a> {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(GeometryFromSql { wkb: raw })
    }

    fn accepts(ty: &Type) -> bool {
        matches!(ty.name(), "geometry" | "geography")
    }
}

/// The projected field a `NUMERIC` result column is read into.
///
/// By position when the projection has exactly one field per result column,
/// and by name otherwise. Every caller derives the statement and the schema
/// from one ordered source — `SqlTable::scan` spells its SELECT list out of the
/// projected fields, and a federated statement is unparsed from the plan whose
/// schema this is — so when the widths agree the columns line up one-to-one.
/// Names do not: the unparser emits a bare call, so Postgres names an aggregate
/// `avg` where the plan names it `avg(hits.UserID)`, and two aggregates over
/// the same function, or a user alias that happens to spell a bare function
/// name (`sum(x) AS avg`), collide on it. A name lookup first would bind the
/// first `avg` to whichever field is *called* `avg`, which is the alias, and
/// decode a fractional average as the alias's `Int64`.
fn projected_destination_field<'a>(
    projected_schema: Option<&'a SchemaRef>,
    column_name: &str,
    column_index: usize,
    column_count: usize,
) -> Option<&'a Field> {
    let schema = projected_schema?;
    if schema.fields().len() == column_count {
        return schema.fields().get(column_index).map(Arc::as_ref);
    }
    schema.field_with_name(column_name).ok()
}

/// A `NUMERIC` value as the exact decimal digits Postgres sent, before any
/// representation narrows them.
///
/// `rust_decimal` holds a 96-bit coefficient, so a value wider than 28 digits
/// is rounded on decode without a word — a `numeric` column holds up to 131072
/// digits before the point, and Postgres's own arithmetic on one (`sum`, a
/// division) settles on whatever scale it needs. Its float conversion is not
/// correctly rounded either. A destination that is not `Decimal128` needs
/// neither loss: `str::parse` rounds the exact digits to the nearest float once,
/// and an integer either fits `i64` or does not.
#[derive(Debug, PartialEq, Eq)]
enum NumericText {
    /// The value's decimal digits, `-` prefixed when negative, with exactly the
    /// `dscale` fractional digits Postgres itself would print.
    Finite(String),
    NaN,
    Infinity {
        negative: bool,
    },
}

/// `NumericText`'s wire encoding: a sign word of `0x4000` is negative, and the
/// three values that carry no digits each have a sign word of their own.
/// `NUMERIC_DSCALE_MASK` in `numeric.c`: the widest `dscale` Postgres itself
/// accepts on receive, and the widest it sends — `round(5::numeric, 16383)`
/// goes out with exactly this scale word, measured on Postgres 16.
const NUMERIC_MAX_DSCALE: u16 = 0x3FFF;
const NUMERIC_SIGN_NEGATIVE: u16 = 0x4000;
const NUMERIC_SIGN_NAN: u16 = 0xC000;
const NUMERIC_SIGN_POSITIVE_INFINITY: u16 = 0xD000;
const NUMERIC_SIGN_NEGATIVE_INFINITY: u16 = 0xF000;

impl<'a> FromSql<'a> for NumericText {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        decode_numeric_wire(raw).map_err(Into::into)
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }
}

/// Decodes Postgres's binary `NUMERIC` (`numeric_send`): `ndigits`, `weight`,
/// `sign` and `dscale` as 16-bit words, then `ndigits` base-10000 digits, the
/// first of which is the coefficient of `10000^weight`.
fn decode_numeric_wire(raw: &[u8]) -> std::result::Result<NumericText, String> {
    let mut cursor = std::io::Cursor::new(raw);
    let truncated = |what: &str| format!("the NUMERIC value on the wire ends before its {what}");
    let ndigits = cursor
        .read_u16::<BigEndian>()
        .map_err(|_| truncated("digit count"))?;
    // Signed: a value below 1 has a negative weight.
    let weight = cursor
        .read_i16::<BigEndian>()
        .map_err(|_| truncated("weight"))?;
    let sign = cursor
        .read_u16::<BigEndian>()
        .map_err(|_| truncated("sign"))?;
    let dscale = cursor
        .read_u16::<BigEndian>()
        .map_err(|_| truncated("scale"))?;
    // `numeric_recv` refuses the same, so a wider scale is not a value Postgres
    // sent but a malformed one — and it bounds the text rendered per value.
    if dscale > NUMERIC_MAX_DSCALE {
        return Err(format!(
            "the NUMERIC value on the wire has a display scale of {dscale}, past the {NUMERIC_MAX_DSCALE} Postgres allows"
        ));
    }

    let negative = match sign {
        0 => false,
        NUMERIC_SIGN_NEGATIVE => true,
        NUMERIC_SIGN_NAN => return Ok(NumericText::NaN),
        NUMERIC_SIGN_POSITIVE_INFINITY => return Ok(NumericText::Infinity { negative: false }),
        NUMERIC_SIGN_NEGATIVE_INFINITY => return Ok(NumericText::Infinity { negative: true }),
        other => {
            return Err(format!(
                "the NUMERIC value on the wire has an unknown sign word {other:#06x}"
            ))
        }
    };

    let mut digits = Vec::with_capacity(usize::from(ndigits));
    for _ in 0..ndigits {
        let digit = cursor
            .read_u16::<BigEndian>()
            .map_err(|_| truncated("digits"))?;
        if digit > 9999 {
            return Err(format!(
                "the NUMERIC value on the wire has a base-10000 digit of {digit}"
            ));
        }
        digits.push(digit);
    }

    Ok(NumericText::Finite(render_numeric_text(
        negative, weight, dscale, &digits,
    )))
}

/// Renders the decoded groups the way Postgres's own text output does: every
/// integer group after the first zero-padded to four digits, and exactly
/// `dscale` fractional digits, zero-filled past the last stored group.
fn render_numeric_text(negative: bool, weight: i16, dscale: u16, digits: &[u16]) -> String {
    // Groups the value does not store are zero: leading ones before the first
    // stored group when `weight` is negative, trailing ones Postgres trimmed.
    let group = |index: i32| -> u16 {
        usize::try_from(index)
            .ok()
            .and_then(|index| digits.get(index).copied())
            .unwrap_or(0)
    };
    let push_group = |text: &mut String, value: u16, padded: bool| {
        let rendered = value.to_string();
        if padded {
            text.extend(std::iter::repeat_n(
                '0',
                4_usize.saturating_sub(rendered.len()),
            ));
        }
        text.push_str(&rendered);
    };

    let dscale = usize::from(dscale);
    let mut text = String::with_capacity(digits.len() * 4 + dscale + 3);
    if negative {
        text.push('-');
    }
    if weight < 0 {
        text.push('0');
    } else {
        for index in 0..=i32::from(weight) {
            push_group(&mut text, group(index), index != 0);
        }
    }
    if dscale > 0 {
        text.push('.');
        let mut fraction = String::with_capacity(dscale + 4);
        let mut index = i32::from(weight) + 1;
        while fraction.len() < dscale {
            push_group(&mut fraction, group(index), true);
            index += 1;
        }
        fraction.truncate(dscale);
        text.push_str(&fraction);
    }
    text
}

impl NumericText {
    /// The nearest `f64`, or `None` for a finite value beyond the type's range.
    /// Rounding happens once, from the exact digits.
    fn to_f64(&self) -> Option<f64> {
        match self {
            NumericText::Finite(text) => text.parse::<f64>().ok().filter(|v| v.is_finite()),
            NumericText::NaN => Some(f64::NAN),
            NumericText::Infinity { negative: false } => Some(f64::INFINITY),
            NumericText::Infinity { negative: true } => Some(f64::NEG_INFINITY),
        }
    }

    /// The nearest `f32`, rounded once from the exact digits rather than
    /// through `f64`.
    fn to_f32(&self) -> Option<f32> {
        match self {
            NumericText::Finite(text) => text.parse::<f32>().ok().filter(|v| v.is_finite()),
            NumericText::NaN => Some(f32::NAN),
            NumericText::Infinity { negative: false } => Some(f32::INFINITY),
            NumericText::Infinity { negative: true } => Some(f32::NEG_INFINITY),
        }
    }

    /// The value truncated toward zero as `i64`, or `None` when its integer
    /// part is out of range or it is not a number at all.
    ///
    /// An `Int64` destination is DataFusion's type for integer arithmetic it
    /// pushed down — a bare `sum` over integers, but also `sum(x) / count(*)`,
    /// which DataFusion evaluates as integer division and Postgres answers as
    /// the exact `numeric` `1.5`. Truncating toward zero is what integer
    /// division does with that quotient, and what the `Decimal128` → `Int64`
    /// cast this read replaces did with it, so `1.5` reads as `1` and `-1.5`
    /// as `-1`. Only a value past `i64` is refused, as the local aggregate
    /// would refuse it.
    fn to_i64(&self) -> Option<i64> {
        let NumericText::Finite(text) = self else {
            return None;
        };
        let integer = text
            .split_once('.')
            .map_or(text.as_str(), |(integer, _)| integer);
        integer.parse::<i64>().ok()
    }
}

impl std::fmt::Display for NumericText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericText::Finite(text) => f.write_str(text),
            NumericText::NaN => f.write_str("NaN"),
            NumericText::Infinity { negative: false } => f.write_str("Infinity"),
            NumericText::Infinity { negative: true } => f.write_str("-Infinity"),
        }
    }
}

/// Appends a `NUMERIC` value to the builder of a `Float64`, `Float32` or
/// `Int64` destination field.
fn append_numeric_to_destination(
    builder: &mut Option<Box<dyn ArrayBuilder>>,
    index: usize,
    field: &Field,
    value: Option<NumericText>,
) -> Result<()> {
    let Some(builder) = builder else {
        return NoBuilderForIndexSnafu { index }.fail();
    };
    let not_representable = |value: &NumericText| {
        NumericNotRepresentableSnafu {
            column: field.name().clone(),
            value: value.to_string(),
            target: field.data_type().clone(),
        }
        .build()
    };
    match field.data_type() {
        DataType::Float64 => {
            let Some(builder) = builder.as_any_mut().downcast_mut::<Float64Builder>() else {
                return FailedToDowncastBuilderSnafu {
                    postgres_type: format!("{}", Type::NUMERIC),
                }
                .fail();
            };
            match value {
                Some(value) => {
                    builder.append_value(value.to_f64().ok_or_else(|| not_representable(&value))?)
                }
                None => builder.append_null(),
            }
        }
        DataType::Float32 => {
            let Some(builder) = builder.as_any_mut().downcast_mut::<Float32Builder>() else {
                return FailedToDowncastBuilderSnafu {
                    postgres_type: format!("{}", Type::NUMERIC),
                }
                .fail();
            };
            match value {
                Some(value) => {
                    builder.append_value(value.to_f32().ok_or_else(|| not_representable(&value))?)
                }
                None => builder.append_null(),
            }
        }
        DataType::Int64 => {
            let Some(builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() else {
                return FailedToDowncastBuilderSnafu {
                    postgres_type: format!("{}", Type::NUMERIC),
                }
                .fail();
            };
            match value {
                Some(value) => {
                    builder.append_value(value.to_i64().ok_or_else(|| not_representable(&value))?)
                }
                None => builder.append_null(),
            }
        }
        DataType::Decimal128(precision, scale) => {
            let Some(builder) = builder.as_any_mut().downcast_mut::<Decimal128Builder>() else {
                return FailedToDowncastBuilderSnafu {
                    postgres_type: format!("{}", Type::NUMERIC),
                }
                .fail();
            };
            match value {
                Some(NumericText::Finite(text)) => {
                    let coefficient = numeric_text_coefficient(&text, *precision, *scale).map_err(
                        |NumericFit::PrecisionTooNarrow| {
                            NumericValueTooLargeSnafu {
                                column: field.name().clone(),
                                column_precision: *precision,
                                column_scale: *scale,
                            }
                            .build()
                        },
                    )?;
                    builder.append_value(coefficient);
                }
                Some(value) => return Err(not_representable(&value)),
                None => builder.append_null(),
            }
        }
        other => {
            return FailedToDowncastBuilderSnafu {
                postgres_type: format!("{} read as {other}", Type::NUMERIC),
            }
            .fail();
        }
    }
    Ok(())
}

/// The `Decimal128` coefficient of a finite `NumericText` at `precision` and
/// `scale`, or why it does not fit them.
///
/// Widening to the column's scale is exact (it only appends zeros). Narrowing
/// rounds half away from zero at the exact digits, which reproduces what
/// casting the value to `NUMERIC(precision, scale)` at the source would have
/// produced — and narrowing is routine: federation pushes an aggregate or a
/// division to Postgres, whose own `NUMERIC` arithmetic settles on a scale of
/// its own, often wider than the one the plan committed to (`AVG` on a
/// `NUMERIC(15,2)` column is planned as `Decimal128(38, 6)`, while Postgres
/// computes it to 16 places). A coefficient of more than `precision` digits is
/// refused. Working from the digits rather than a `rust_decimal::Decimal` means
/// a value wider than that type's 28-digit coefficient — a 37-digit `sum` over
/// a `numeric` column, say — lands on a `Decimal128(38, s)` it fits with every
/// digit intact instead of being rounded on decode without a word.
fn numeric_text_coefficient(text: &str, precision: u8, scale: i8) -> Result<i128, NumericFit> {
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));

    // The digit string of the coefficient at `scale`, and the digits dropped
    // below it — the first of those decides the rounding.
    let (kept, dropped): (String, String) = if scale >= 0 {
        let scale = usize::from(scale.unsigned_abs());
        if fraction.len() <= scale {
            (
                format!("{integer}{fraction}{}", "0".repeat(scale - fraction.len())),
                String::new(),
            )
        } else {
            (
                format!("{integer}{}", &fraction[..scale]),
                fraction[scale..].to_string(),
            )
        }
    } else {
        // A negative scale counts trailing integer digits the coefficient
        // does not carry: `NUMERIC(2, -3)` stores `12000` as `12`.
        let drop = usize::from(scale.unsigned_abs());
        if integer.len() <= drop {
            // Every integer digit is dropped; the ones the value does not
            // spell are zeros, and they come first — `50` into `(2, -3)` drops
            // `050`, whose first digit decides the rounding, not the `5`.
            (String::new(), format!("{integer:0>drop$}{fraction}"))
        } else {
            (
                integer[..integer.len() - drop].to_string(),
                format!("{}{fraction}", &integer[integer.len() - drop..]),
            )
        }
    };

    let kept = kept.trim_start_matches('0');
    let mut coefficient: i128 = if kept.is_empty() {
        0
    } else {
        kept.parse().map_err(|_| NumericFit::PrecisionTooNarrow)?
    };
    let round_away_from_zero = dropped
        .as_bytes()
        .first()
        .is_some_and(|digit| *digit >= b'5');
    if round_away_from_zero {
        coefficient = coefficient
            .checked_add(1)
            .ok_or(NumericFit::PrecisionTooNarrow)?;
    }

    let limit = 10u128
        .checked_pow(u32::from(precision))
        .ok_or(NumericFit::PrecisionTooNarrow)?;
    if coefficient.unsigned_abs() >= limit {
        return Err(NumericFit::PrecisionTooNarrow);
    }
    Ok(if negative { -coefficient } else { coefficient })
}

fn get_decimal_array_column_precision_and_scale(
    column_name: &str,
    projected_schema: &SchemaRef,
) -> Option<(u8, i8)> {
    let field = projected_schema.field_with_name(column_name).ok()?;
    match field.data_type() {
        DataType::List(inner_field) | DataType::LargeList(inner_field) => {
            match inner_field.data_type() {
                DataType::Decimal128(precision, scale) => Some((*precision, *scale)),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveTime;
    use datafusion::arrow::array::{
        Array, ListArray, StringArray, StructArray, Time64NanosecondArray, Time64NanosecondBuilder,
    };
    use geo_types::{point, polygon, Geometry};
    use geozero::{CoordDimensions, ToWkb};
    use std::str::FromStr;

    /// 1969-12-31 23:59:59.5 is half a second before the epoch, not an error and not a
    /// skipped value.
    #[test]
    fn unix_nanos_is_negative_before_the_epoch() {
        let half = std::time::Duration::from_millis(500);
        assert_eq!(unix_nanos(UNIX_EPOCH - half).expect("before"), -500_000_000);
        assert_eq!(unix_nanos(UNIX_EPOCH).expect("epoch"), 0);
        assert_eq!(unix_nanos(UNIX_EPOCH + half).expect("after"), 500_000_000);
    }

    /// Big-endian 16-bit words, the way `numeric_send` lays a value out.
    fn numeric_wire(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_be_bytes()).collect()
    }

    #[test]
    fn numeric_text_renders_the_digits_postgres_would_print() {
        // (ndigits, weight, sign, dscale, digits...) and the text Postgres prints for it.
        let cases: &[(&[u16], &str)] = &[
            (&[0, 0, 0, 0], "0"),
            (&[1, 0, 0, 0, 1], "1"),
            (&[1, 0, NUMERIC_SIGN_NEGATIVE, 1, 1], "-1.0"),
            // A weight of -1: the first stored group is the first fractional one.
            (&[1, 0xFFFF, 0, 4, 1], "0.0001"),
            (&[1, 0xFFFE, 0, 8, 1], "0.00000001"),
            // Trailing zero groups Postgres trimmed are zero-filled back in.
            (&[1, 2, 0, 0, 1], "100000000"),
            (&[2, 1, 0, 2, 1, 5], "10005.00"),
            // An inner group below 1000 is zero-padded; the leading one is not.
            (&[3, 1, 0, 1, 1234, 5678, 9000], "12345678.9"),
            (&[3, 2, 0, 0, 1, 42, 7], "100420007"),
            // Fewer fractional digits printed than stored groups carry.
            (&[2, 0, 0, 2, 5, 1234], "5.12"),
            // 35 significant digits, past what `rust_decimal` can hold.
            (
                &[
                    9, 4, 0, 16, 252, 8953, 297, 8971, 5791, 8666, 6666, 6666, 6667,
                ],
                "2528953029789715791.8666666666666667",
            ),
        ];
        for (words, expected) in cases {
            let decoded = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(words))
                .expect("a well-formed NUMERIC decodes");
            assert_eq!(
                decoded,
                NumericText::Finite((*expected).to_string()),
                "words {words:?}"
            );
        }
    }

    #[test]
    fn numeric_text_decodes_the_three_special_values() {
        for (sign, expected) in [
            (NUMERIC_SIGN_NAN, NumericText::NaN),
            (
                NUMERIC_SIGN_POSITIVE_INFINITY,
                NumericText::Infinity { negative: false },
            ),
            (
                NUMERIC_SIGN_NEGATIVE_INFINITY,
                NumericText::Infinity { negative: true },
            ),
        ] {
            let decoded = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[0, 0, sign, 0]))
                .expect("a special NUMERIC decodes");
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn numeric_text_refuses_a_malformed_wire_value() {
        let truncated = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[1, 0, 0]))
            .expect_err("a header cut short is refused");
        assert!(
            truncated.to_string().contains("ends before its scale"),
            "{truncated}"
        );
        let missing_digit = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[2, 0, 0, 0, 1]))
            .expect_err("fewer digits than announced is refused");
        assert!(
            missing_digit.to_string().contains("ends before its digits"),
            "{missing_digit}"
        );
        let bad_sign = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[0, 0, 0x1234, 0]))
            .expect_err("an unknown sign word is refused");
        assert!(bad_sign.to_string().contains("0x1234"), "{bad_sign}");
        let bad_digit = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[1, 0, 0, 0, 10000]))
            .expect_err("a base-10000 digit of 10000 is refused");
        assert!(bad_digit.to_string().contains("10000"), "{bad_digit}");
    }

    #[test]
    fn numeric_text_keeps_every_digit_rust_decimal_would_round_away() {
        let wire = numeric_wire(&[
            9, 4, 0, 16, 252, 8953, 297, 8971, 5791, 8666, 6666, 6666, 6667,
        ]);
        let text = NumericText::from_sql(&Type::NUMERIC, &wire).expect("decodes");
        // 35 significant digits: past `rust_decimal`'s 28, which rounds the value
        // to 2528953029789715791.866666667 on decode.
        let exact = "2528953029789715791.8666666666666667";
        let through_rust_decimal = Decimal::from_sql(&Type::NUMERIC, &wire)
            .expect("decodes")
            .to_string();
        assert!(
            through_rust_decimal != exact
                && through_rust_decimal.starts_with("2528953029789715791.8666")
                && through_rust_decimal.len() < exact.len(),
            "rust_decimal rounds the value to its 28-digit coefficient: {through_rust_decimal}"
        );
        assert_eq!(text, NumericText::Finite(exact.to_string()));
        assert_eq!(
            text.to_f64().expect("finite"),
            exact.parse::<f64>().expect("parses"),
            "one correctly rounded conversion from the exact digits"
        );
    }

    #[test]
    fn numeric_text_to_i64_truncates_toward_zero_within_range() {
        let finite = |text: &str| NumericText::Finite(text.to_string());
        assert_eq!(finite("5.000").to_i64(), Some(5));
        assert_eq!(finite("-0").to_i64(), Some(0));
        // `sum(x) / count(*)` pushed down: integer division's answer, not floor's.
        assert_eq!(finite("1.5").to_i64(), Some(1));
        assert_eq!(finite("-1.5").to_i64(), Some(-1));
        assert_eq!(finite("-0.9").to_i64(), Some(0));
        assert_eq!(finite("0.9999").to_i64(), Some(0));
        assert_eq!(finite("9223372036854775807").to_i64(), Some(i64::MAX));
        assert_eq!(finite("9223372036854775808").to_i64(), None);
        assert_eq!(finite("-9223372036854775808").to_i64(), Some(i64::MIN));
        assert_eq!(NumericText::NaN.to_i64(), None);
        assert_eq!(NumericText::Infinity { negative: false }.to_i64(), None);
    }

    #[test]
    fn numeric_text_float_conversions_round_once_from_the_digits() {
        let finite = |text: &str| NumericText::Finite(text.to_string());
        assert_eq!(finite("47.5").to_f64(), Some(47.5));
        assert_eq!(finite("0.1").to_f64(), Some(0.1));
        assert_eq!(finite("0.1").to_f32(), Some(0.1_f32));
        // Rounded once from the digits rather than through f64 first, so it agrees
        // with the standard library's own decimal-to-f32 parse on a value that
        // carries more digits than either float holds.
        let long = "1.00000005960464477539062500001";
        assert_eq!(
            finite(long).to_f32(),
            Some(long.parse::<f32>().expect("parses"))
        );
        let huge = finite(&format!("1{}", "0".repeat(400)));
        assert_eq!(
            huge.to_f64(),
            None,
            "beyond f64's range is not representable"
        );
        assert!(NumericText::NaN.to_f64().expect("NaN maps").is_nan());
        assert_eq!(
            NumericText::Infinity { negative: true }.to_f64(),
            Some(f64::NEG_INFINITY)
        );
    }

    #[test]
    fn numeric_text_coefficient_widens_exactly_and_narrows_half_away_from_zero() {
        // Widening only appends zeros.
        assert_eq!(numeric_text_coefficient("1.5", 38, 6), Ok(1_500_000));
        assert_eq!(numeric_text_coefficient("-1.5", 38, 6), Ok(-1_500_000));
        assert_eq!(
            numeric_text_coefficient("1000000000", 38, 20),
            Ok(100_000_000_000_000_000_000_000_000_000)
        );
        // Narrowing rounds half away from zero, on both sides of zero.
        assert_eq!(
            numeric_text_coefficient("1.6666666666666667", 38, 6),
            Ok(1_666_667)
        );
        assert_eq!(numeric_text_coefficient("1.2345", 38, 2), Ok(123));
        assert_eq!(numeric_text_coefficient("1.2350", 38, 2), Ok(124));
        assert_eq!(numeric_text_coefficient("-1.2350", 38, 2), Ok(-124));
        assert_eq!(numeric_text_coefficient("0.0049", 38, 2), Ok(0));
        assert_eq!(numeric_text_coefficient("-0.0050", 38, 2), Ok(-1));
        // A carry can add a digit.
        assert_eq!(numeric_text_coefficient("9.9999", 38, 2), Ok(1000));
        // Zero, however spelled.
        assert_eq!(numeric_text_coefficient("0", 38, 20), Ok(0));
        assert_eq!(numeric_text_coefficient("-0.000", 38, 2), Ok(0));
    }

    #[test]
    fn numeric_text_coefficient_handles_a_negative_scale() {
        // `NUMERIC(2, -3)` stores 12000 as the coefficient 12.
        assert_eq!(numeric_text_coefficient("12000", 2, -3), Ok(12));
        assert_eq!(numeric_text_coefficient("12500", 2, -3), Ok(13));
        assert_eq!(numeric_text_coefficient("12499.9", 2, -3), Ok(12));
        assert_eq!(numeric_text_coefficient("-12500", 2, -3), Ok(-13));
        assert_eq!(numeric_text_coefficient("400", 2, -3), Ok(0));
        assert_eq!(numeric_text_coefficient("500", 2, -3), Ok(1));
        // Fewer integer digits than the scale drops: the missing ones are
        // leading zeros, so 50 is nowhere near the 500 that would round up.
        assert_eq!(numeric_text_coefficient("50", 2, -3), Ok(0));
        assert_eq!(numeric_text_coefficient("-50", 2, -3), Ok(0));
        assert_eq!(numeric_text_coefficient("499.9", 2, -3), Ok(0));
        assert_eq!(numeric_text_coefficient("5", 2, -1), Ok(1));
        assert_eq!(numeric_text_coefficient("4.9", 2, -1), Ok(0));
    }

    #[test]
    fn numeric_text_coefficient_refuses_what_the_precision_cannot_hold() {
        // 19 integer digits at scale 20 need 39: the issue's failure, still refused
        // where the plan really asks for Decimal128(38, 20).
        assert!(matches!(
            numeric_text_coefficient("2528953029789715791", 38, 20),
            Err(NumericFit::PrecisionTooNarrow)
        ));
        // 18 integer digits fit.
        assert_eq!(
            numeric_text_coefficient("252895302978971580", 38, 20),
            Ok(25_289_530_297_897_158_000_000_000_000_000_000_000)
        );
        // `precision` digits fit; one more does not — including by carry.
        assert!(numeric_text_coefficient(&"9".repeat(38), 38, 0).is_ok());
        assert!(matches!(
            numeric_text_coefficient(&format!("1{}", "0".repeat(38)), 38, 0),
            Err(NumericFit::PrecisionTooNarrow)
        ));
        assert!(matches!(
            numeric_text_coefficient(&format!("{}.5", "9".repeat(38)), 38, 0),
            Err(NumericFit::PrecisionTooNarrow)
        ));
        assert_eq!(
            numeric_text_coefficient(&format!("{}.5", "9".repeat(37)), 38, 0),
            Ok(10_i128.pow(37))
        );
    }

    #[test]
    fn numeric_text_coefficient_keeps_every_digit_of_a_wide_value() {
        // 35 significant digits into Decimal128(38, 16): exact, where `rust_decimal`
        // would have rounded the value on decode.
        assert_eq!(
            numeric_text_coefficient("2528953029789715791.8666666666666667", 38, 16),
            Ok(25_289_530_297_897_157_918_666_666_666_666_667)
        );
        // 37 digits into Decimal128(38, 12).
        assert_eq!(
            numeric_text_coefficient("1234567890123456789012345.123456789013", 38, 12),
            Ok(1_234_567_890_123_456_789_012_345_123_456_789_013)
        );
    }

    #[test]
    fn duplicate_result_names_keep_their_positional_nullability() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("id", DataType::Int32, true),
        ]));
        assert_eq!(
            projected_destination_field(Some(&schema), "id", 0, 2).map(Field::is_nullable),
            Some(false)
        );
        assert_eq!(
            projected_destination_field(Some(&schema), "id", 1, 2).map(Field::is_nullable),
            Some(true)
        );
    }

    #[test]
    fn projected_destination_field_matches_by_position_when_the_widths_agree() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("avg(hits.UserID)", DataType::Float64, true),
            Field::new("sum(hits.UserID)", DataType::Int64, true),
        ]));
        // A bare `avg`/`sum` the unparser emitted is matched to its position.
        assert_eq!(
            projected_destination_field(Some(&schema), "avg", 0, 2).map(Field::data_type),
            Some(&DataType::Float64)
        );
        assert_eq!(
            projected_destination_field(Some(&schema), "sum", 1, 2).map(Field::data_type),
            Some(&DataType::Int64)
        );
        // No positional match when the projection is not one field per column;
        // the name decides, and a name the projection lacks resolves to nothing.
        assert_eq!(
            projected_destination_field(Some(&schema), "sum(hits.UserID)", 0, 3)
                .map(Field::data_type),
            Some(&DataType::Int64)
        );
        assert_eq!(
            projected_destination_field(Some(&schema), "sum", 1, 3),
            None
        );
        assert_eq!(projected_destination_field(None, "sum", 1, 2), None);
    }

    #[test]
    fn projected_destination_field_is_not_captured_by_a_colliding_alias() {
        // `SELECT avg(x), sum(x) AS avg`: Postgres names both columns `avg`, and the
        // second plan field is literally called `avg`. Position keeps the first
        // column on the Float64 the average needs; a name lookup would have bound
        // it to the alias's Int64 and refused 1.5.
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("avg(t.x)", DataType::Float64, true),
            Field::new("avg", DataType::Int64, true),
        ]));
        assert_eq!(
            projected_destination_field(Some(&schema), "avg", 0, 2).map(Field::name),
            Some(&"avg(t.x)".to_string())
        );
        assert_eq!(
            projected_destination_field(Some(&schema), "avg", 1, 2).map(Field::name),
            Some(&"avg".to_string())
        );
    }

    #[test]
    fn numeric_text_refuses_a_display_scale_postgres_would_not_send() {
        // `round(5::numeric, 16383)` really does arrive with this scale word.
        let accepted = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[1, 0, 0, 0x3FFF, 5]))
            .expect("the widest scale Postgres allows decodes");
        assert_eq!(
            accepted,
            NumericText::Finite(format!("5.{}", "0".repeat(0x3FFF)))
        );
        assert_eq!(accepted.to_i64(), Some(5));
        assert_eq!(accepted.to_f64(), Some(5.0));
        let refused = NumericText::from_sql(&Type::NUMERIC, &numeric_wire(&[1, 0, 0, 0x4000, 5]))
            .expect_err("a scale past NUMERIC_DSCALE_MASK is refused");
        assert!(refused.to_string().contains("16384"), "{refused}");
    }

    #[allow(clippy::cast_possible_truncation)]
    #[tokio::test]
    async fn test_decimal_from_sql() {
        let positive_u16: Vec<u16> = vec![5, 3, 0, 5, 9345, 1293, 2903, 1293, 932];
        let positive_raw: Vec<u8> = positive_u16
            .iter()
            .flat_map(|&x| vec![(x >> 8) as u8, x as u8])
            .collect();
        let positive = Decimal::from_str("9345129329031293.0932").expect("Failed to parse decimal");
        let positive_result = Decimal::from_sql(&Type::NUMERIC, positive_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(positive_result, positive);

        let negative_u16: Vec<u16> = vec![5, 3, 0x4000, 5, 9345, 1293, 2903, 1293, 932];
        let negative_raw: Vec<u8> = negative_u16
            .iter()
            .flat_map(|&x| vec![(x >> 8) as u8, x as u8])
            .collect();

        let negative =
            Decimal::from_str("-9345129329031293.0932").expect("Failed to parse decimal");
        let negative_result = Decimal::from_sql(&Type::NUMERIC, negative_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(negative_result, negative);
    }

    #[test]
    fn test_interval_from_sql() {
        let positive_time: i64 = 123_123;
        let positive_day: i32 = 10;
        let positive_month: i32 = 2;

        let mut positive_raw: Vec<u8> = Vec::new();
        positive_raw.extend_from_slice(&positive_time.to_be_bytes());
        positive_raw.extend_from_slice(&positive_day.to_be_bytes());
        positive_raw.extend_from_slice(&positive_month.to_be_bytes());

        let positive_result = IntervalFromSql::from_sql(&Type::INTERVAL, positive_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(positive_result.day, positive_day);
        assert_eq!(positive_result.time, positive_time);
        assert_eq!(positive_result.month, positive_month);

        let negative_time: i64 = -123_123;
        let negative_day: i32 = -10;
        let negative_month: i32 = -2;

        let mut negative_raw: Vec<u8> = Vec::new();
        negative_raw.extend_from_slice(&negative_time.to_be_bytes());
        negative_raw.extend_from_slice(&negative_day.to_be_bytes());
        negative_raw.extend_from_slice(&negative_month.to_be_bytes());

        let negative_result = IntervalFromSql::from_sql(&Type::INTERVAL, negative_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(negative_result.day, negative_day);
        assert_eq!(negative_result.time, negative_time);
        assert_eq!(negative_result.month, negative_month);
    }

    #[test]
    fn test_money_from_sql() {
        let positive_cash_value: i64 = 123;
        let mut positive_raw: Vec<u8> = Vec::new();
        positive_raw.extend_from_slice(&positive_cash_value.to_be_bytes());

        let positive_result = MoneyFromSql::from_sql(&Type::MONEY, positive_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(positive_result.cash_value, positive_cash_value);

        let negative_cash_value: i64 = -123;
        let mut negative_raw: Vec<u8> = Vec::new();
        negative_raw.extend_from_slice(&negative_cash_value.to_be_bytes());

        let negative_result = MoneyFromSql::from_sql(&Type::MONEY, negative_raw.as_slice())
            .expect("Failed to run FromSql");
        assert_eq!(negative_result.cash_value, negative_cash_value);
    }

    #[test]
    fn test_chrono_naive_time_to_time64nanosecond() {
        let chrono_naive_vec = vec![
            NaiveTime::from_hms_opt(10, 30, 00).unwrap_or_default(),
            NaiveTime::from_hms_opt(10, 45, 15).unwrap_or_default(),
        ];

        let time_array: Time64NanosecondArray = vec![
            (10 * 3600 + 30 * 60) * 1_000_000_000,
            (10 * 3600 + 45 * 60 + 15) * 1_000_000_000,
        ]
        .into();

        let mut builder = Time64NanosecondBuilder::new();
        for time in chrono_naive_vec {
            let timestamp: i64 = i64::from(time.num_seconds_from_midnight()) * 1_000_000_000
                + i64::from(time.nanosecond());
            builder.append_value(timestamp);
        }
        let converted_result = builder.finish();
        assert_eq!(converted_result, time_array);
    }

    #[test]
    fn test_geometry_from_sql() {
        let positive_geometry = Geometry::from(point! { x: 181.2, y: 51.79 })
            .to_wkb(CoordDimensions::xy())
            .unwrap();
        let mut positive_raw: Vec<u8> = Vec::new();
        positive_raw.extend_from_slice(&positive_geometry);

        let positive_result = GeometryFromSql::from_sql(
            &Type::new(
                "geometry".to_owned(),
                16462,
                Kind::Simple,
                "public".to_owned(),
            ),
            positive_raw.as_slice(),
        )
        .expect("Failed to run FromSql");
        assert_eq!(positive_result.wkb, positive_geometry);

        let positive_geometry = Geometry::from(polygon![
            (x: -111., y: 45.),
            (x: -111., y: 41.),
            (x: -104., y: 41.),
            (x: -104., y: 45.),
        ])
        .to_wkb(CoordDimensions::xy())
        .unwrap();
        let mut positive_raw: Vec<u8> = Vec::new();
        positive_raw.extend_from_slice(&positive_geometry);

        let positive_result = GeometryFromSql::from_sql(
            &Type::new(
                "geometry".to_owned(),
                16462,
                Kind::Simple,
                "public".to_owned(),
            ),
            positive_raw.as_slice(),
        )
        .expect("Failed to run FromSql");
        assert_eq!(positive_result.wkb, positive_geometry);
    }

    #[test]
    fn test_jsonb_raw_string_from_sql() {
        // JSONB happy path: version byte 0x01 is stripped
        let json = r#"{"key":"value"}"#;
        let mut jsonb_raw: Vec<u8> = vec![0x01];
        jsonb_raw.extend_from_slice(json.as_bytes());
        let result = JsonbRawString::from_sql(&Type::JSONB, &jsonb_raw)
            .expect("Failed to run FromSql for JSONB");
        assert_eq!(result.0, json);

        // JSON happy path: bytes returned as-is (no version byte)
        let json_raw = json.as_bytes();
        let result = JsonbRawString::from_sql(&Type::JSON, json_raw)
            .expect("Failed to run FromSql for JSON");
        assert_eq!(result.0, json);

        // JSONB wrong version byte → error
        let err = JsonbRawString::from_sql(&Type::JSONB, &[0x02, b'{', b'}'])
            .expect_err("Expected error for wrong JSONB version");
        assert!(
            err.to_string()
                .contains("unsupported JSONB encoding version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_decode_json_list_of_struct_user_shape() {
        let string_array = StringArray::from(vec![
            Some(
                r#"[{"id":"u1","email":"test@doss-sql.test","first_name":"Test","last_name":"User"}]"#,
            ),
            Some("[]"),
            None,
        ]);

        let list_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Field::new("id", DataType::Utf8, true),
                    Field::new("email", DataType::Utf8, true),
                    Field::new("first_name", DataType::Utf8, true),
                    Field::new("last_name", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        ));

        let array = decode_json_complex_column(
            &string_array,
            &Field::new_list("item", Arc::clone(&list_item_field), true),
        )
        .expect("cast succeeds");
        let list = array
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("array should be ListArray");

        assert_eq!(list.len(), 3);
        assert!(!list.is_null(0));
        assert_eq!(list.value_length(0), 1);
        assert!(!list.is_null(1));
        assert_eq!(list.value_length(1), 0);
        assert!(list.is_null(2));

        let values = list.value(0);
        let struct_values = values
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list values should be StructArray");
        let ids = struct_values
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id should be Utf8");
        assert_eq!(ids.value(0), "u1");
    }

    #[test]
    fn test_decode_json_list_of_struct_lookup_value_float() {
        let string_array = StringArray::from(vec![Some(r#"[{"id":"0001","value":30.0}]"#)]);

        let list_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Field::new("id", DataType::Utf8, true),
                    Field::new("value", DataType::Float64, true),
                ]
                .into(),
            ),
            true,
        ));

        let array = decode_json_complex_column(
            &string_array,
            &Field::new_list("item", Arc::clone(&list_item_field), true),
        )
        .expect("cast succeeds");
        let list = array
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("array should be ListArray");
        assert_eq!(list.value_length(0), 1);

        let values = list.value(0);
        let struct_values = values
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list values should be StructArray");
        let float_values = struct_values
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("value should be Float64");
        assert_eq!(float_values.value(0), 30.0);
    }

    #[test]
    fn test_decode_json_list_of_struct_invalid_json_errors() {
        let string_array = StringArray::from(vec![Some("not-json")]);

        let list_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(vec![Field::new("id", DataType::Utf8, true)].into()),
            true,
        ));

        let error = decode_json_complex_column(
            &string_array,
            &Field::new_list("item", Arc::clone(&list_item_field), true),
        )
        .expect_err("malformed json should error");
        assert!(error.to_string().contains("Failed to decode value"));
    }

    #[test]
    fn test_decode_json_list_of_struct_all_null_fast_path() {
        let string_array = StringArray::from(vec![None::<&str>, None, None]);

        let list_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(vec![Field::new("id", DataType::Utf8, true)].into()),
            true,
        ));

        let array = decode_json_complex_column(
            &string_array,
            &Field::new_list("item", Arc::clone(&list_item_field), true),
        )
        .expect("cast succeeds");
        let list = array
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("array should be ListArray");

        assert_eq!(list.len(), 3);
        assert_eq!(list.null_count(), 3);
        assert!(list.is_null(0));
        assert!(list.is_null(1));
        assert!(list.is_null(2));
    }

    /// Regression guard: the batch NDJSON decode strategy relies on `arrow_json`
    /// treating the JSON literal `null` as a null List entry for nullable List
    /// fields. This test asserts that contract so any future arrow-json upgrade
    /// that changes the behaviour is caught immediately.
    #[test]
    fn test_decode_json_list_of_struct_null_semantics() {
        let string_array = StringArray::from(vec![
            Some(r#"[{"id":"a","value":1.0}]"#),
            None,
            Some("[]"),
            Some(r#"[{"id":"b","value":2.0}]"#),
        ]);

        let list_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Field::new("id", DataType::Utf8, true),
                    Field::new("value", DataType::Float64, true),
                ]
                .into(),
            ),
            true,
        ));

        let array = decode_json_complex_column(
            &string_array,
            &Field::new_list("item", Arc::clone(&list_item_field), true),
        )
        .expect("decode succeeds");
        let list = array
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("array should be ListArray");

        assert_eq!(list.len(), 4);

        // row 0: non-null, length 1
        assert!(!list.is_null(0));
        assert_eq!(list.value_length(0), 1);

        // row 1: NULL (is_null == true)
        assert!(list.is_null(1));

        // row 2: non-null, length 0 (empty list, not null)
        assert!(!list.is_null(2));
        assert_eq!(list.value_length(2), 0);

        // row 3: non-null, length 1
        assert!(!list.is_null(3));
        assert_eq!(list.value_length(3), 1);

        // Verify struct contents of row 3
        let values = list.value(3);
        let struct_values = values
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list values should be StructArray");
        let ids = struct_values
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id should be Utf8");
        assert_eq!(ids.value(0), "b");
        let floats = struct_values
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("value should be Float64");
        assert_eq!(floats.value(0), 2.0);
    }

    /// Redshift Spectrum serializes a top-level STRUCT column to a JSON object
    /// (`{"given":"John","family":"Smith"}`); the row path must decode it into a
    /// `StructArray`, not flatten it into separate columns.
    #[test]
    fn test_decode_json_complex_column_struct() {
        let string_array = StringArray::from(vec![
            Some(r#"{"given":"John","family":"Smith"}"#),
            None,
            Some(r#"{"given":"Ada","family":"Lovelace"}"#),
        ]);

        let struct_field = Field::new(
            "name",
            DataType::Struct(
                vec![
                    Field::new("given", DataType::Utf8, true),
                    Field::new("family", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        );

        let array =
            decode_json_complex_column(&string_array, &struct_field).expect("struct decode");
        let structs = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("array should be StructArray");

        assert_eq!(structs.len(), 3);
        assert!(structs.is_null(1));
        let given = structs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("given should be Utf8");
        assert_eq!(given.value(0), "John");
        assert_eq!(given.value(2), "Ada");
    }

    /// Redshift Spectrum serializes a MAP column to a JSON object with string keys;
    /// the row path must decode it into a `MapArray`.
    #[test]
    fn test_decode_json_complex_column_map() {
        let string_array = StringArray::from(vec![Some(r#"{"a":1,"b":2}"#), Some("{}")]);

        let entries = Arc::new(Field::new_struct(
            "entries",
            vec![
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(Field::new("value", DataType::Int32, true)),
            ],
            false,
        ));
        let map_field = Field::new("attrs", DataType::Map(entries, false), true);

        let array = decode_json_complex_column(&string_array, &map_field).expect("map decode");
        let map = array
            .as_any()
            .downcast_ref::<arrow::array::MapArray>()
            .expect("array should be MapArray");

        assert_eq!(map.len(), 2);
        assert_eq!(map.value_length(0), 2);
        assert_eq!(map.value_length(1), 0);
    }

    /// The headline case: an array of structs (`[{...},{...}]`) decodes into
    /// `List<Struct>`, mirroring how Spectrum serializes collection columns.
    #[test]
    fn test_decode_json_complex_column_array_of_struct() {
        let string_array = StringArray::from(vec![Some(
            r#"[{"shipdate":"2018-03-01","price":100.5},{"shipdate":"2018-03-02","price":7.0}]"#,
        )]);

        let list_field = Field::new_list(
            "lines",
            Field::new(
                "item",
                DataType::Struct(
                    vec![
                        Field::new("shipdate", DataType::Utf8, true),
                        Field::new("price", DataType::Float64, true),
                    ]
                    .into(),
                ),
                true,
            ),
            true,
        );

        let array =
            decode_json_complex_column(&string_array, &list_field).expect("array<struct> decode");
        let list = array
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("array should be ListArray");
        assert_eq!(list.value_length(0), 2);

        let structs = list
            .value(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list values should be StructArray")
            .clone();
        let prices = structs
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("price should be Float64");
        assert_eq!(prices.value(0), 100.5);
        assert_eq!(prices.value(1), 7.0);
    }

    /// A text-like wire column (Redshift serializes Spectrum complex types to
    /// `VARCHAR`) projected as a complex Arrow type must be routed through the JSON
    /// decode path, not read as a scalar string.
    #[test]
    fn test_projected_json_complex_field_triggers_for_varchar_struct() {
        let payload_field = Field::new(
            "payload",
            DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into()),
            true,
        );
        let schema = Arc::new(Schema::new(vec![payload_field]));
        let projected_schema = Some(schema);

        assert!(
            projected_json_complex_field(
                projected_destination_field(projected_schema.as_ref(), "payload", 0, 0),
                &Type::VARCHAR
            )
            .is_some(),
            "VARCHAR column projected as Struct should decode as JSON"
        );

        // A plain scalar projection over the same wire column is left as a scalar.
        let scalar_schema = Some(Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::Utf8,
            true,
        )])));
        assert!(
            projected_json_complex_field(
                projected_destination_field(scalar_schema.as_ref(), "payload", 0, 0),
                &Type::VARCHAR
            )
            .is_none(),
            "VARCHAR column projected as Utf8 must stay a scalar string"
        );
    }

    /// A Redshift `super` wire column (how Spectrum serializes `ARRAY`/`STRUCT`/`MAP`
    /// external columns) is matched by type name and routed through the JSON decode path.
    #[test]
    fn test_super_type_routes_through_json_decode() {
        let super_type = Type::new(
            "super".to_owned(),
            4000,
            Kind::Simple,
            "pg_catalog".to_owned(),
        );

        // Without a projected schema it resolves to a plain Utf8 JSON string column.
        assert_eq!(
            map_column_type_to_data_type(&super_type, "c").expect("super maps"),
            Some(DataType::Utf8)
        );

        // Projected as a complex type, it is picked up for JSON decoding.
        let schema = Some(Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into()),
                true,
            ))),
            true,
        )])));
        assert!(
            projected_json_complex_field(
                projected_destination_field(schema.as_ref(), "payload", 0, 0),
                &super_type
            )
            .is_some(),
            "super column projected as List<Struct> should decode as JSON"
        );
    }

    /// A native PostgreSQL array-of-composite wire type (`composite[]`) maps to
    /// `List<Struct>`, and that Arrow type produces a `ListBuilder<StructBuilder>`.
    #[test]
    fn test_composite_array_maps_to_list_struct() {
        use tokio_postgres::types::Field as PgField;

        let composite = Type::new(
            "line_item".to_owned(),
            20_000,
            Kind::Composite(vec![
                PgField::new("sku".to_owned(), Type::TEXT),
                PgField::new("qty".to_owned(), Type::INT4),
                PgField::new("price".to_owned(), Type::FLOAT8),
            ]),
            "public".to_owned(),
        );
        let array_type = Type::new(
            "_line_item".to_owned(),
            20_001,
            Kind::Array(composite),
            "public".to_owned(),
        );

        let dt = map_column_type_to_data_type(&array_type, "items")
            .expect("maps")
            .expect("some data type");
        let expected_item = DataType::Struct(
            vec![
                Field::new("sku", DataType::Utf8, true),
                Field::new("qty", DataType::Int32, true),
                Field::new("price", DataType::Float64, true),
            ]
            .into(),
        );
        assert_eq!(
            dt,
            DataType::List(Arc::new(Field::new("item", expected_item, true)))
        );

        // The builder for this Arrow type must downcast to ListBuilder<StructBuilder>.
        let mut builder = crate::sql::arrow_sql_gen::arrow::map_data_type_to_array_builder(&dt);
        assert!(
            builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<StructBuilder>>()
                .is_some(),
            "List<Struct> must build a ListBuilder<StructBuilder>"
        );
    }

    #[test]
    fn test_super_raw_string_reads_utf8() {
        let super_type = Type::new(
            "super".to_owned(),
            4000,
            Kind::Simple,
            "pg_catalog".to_owned(),
        );
        let raw = br#"[{"a":1},{"a":2}]"#;
        let parsed = SuperRawString::from_sql(&super_type, raw).expect("super decodes");
        assert_eq!(parsed.0, r#"[{"a":1},{"a":2}]"#);
        assert!(SuperRawString::accepts(&super_type));
        assert!(!SuperRawString::accepts(&Type::VARCHAR));
    }

    #[test]
    fn test_projected_json_complex_field_matches_by_name() {
        let other_field = Field::new("other", DataType::Int32, true);
        let payload_field = Field::new(
            "payload",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(vec![Field::new("email", DataType::Utf8, true)].into()),
                true,
            ))),
            true,
        );

        let schema = Arc::new(Schema::new(vec![other_field, payload_field]));
        let projected_schema = Some(schema);

        // A width mismatch retains the name-based destination lookup.
        let resolved = projected_json_complex_field(
            projected_destination_field(projected_schema.as_ref(), "payload", 0, 0),
            &Type::JSONB,
        )
        .expect("field should resolve from projected schema");
        assert_eq!(resolved.name(), "payload");

        let DataType::List(item_field) = resolved.data_type() else {
            panic!("resolved field should be list");
        };
        let DataType::Struct(fields) = item_field.data_type() else {
            panic!("resolved list item should be struct");
        };
        assert_eq!(fields[0].name(), "email");

        // An unknown name cannot resolve through the width-mismatch fallback.
        assert!(projected_json_complex_field(
            projected_destination_field(projected_schema.as_ref(), "no_such_column", 0, 0),
            &Type::JSONB
        )
        .is_none());
    }
}
