-- One row per store-date. Casts + renames only; no business logic here so
-- that every downstream model shares one definition of these columns.
select
    cast(store as integer) as store_id,
    cast(date as date) as sale_date,
    cast(dayofweek as integer) as day_of_week,
    cast(sales as integer) as sales,
    cast(customers as integer) as customers,
    cast(open as boolean) as is_open,
    cast(promo as boolean) as is_promo,
    stateholiday as state_holiday,
    cast(schoolholiday as boolean) as is_school_holiday
from {{ source('raw', 'train') }}
