-- Grain: one row per store-date. This is the table Phase 2 feature
-- engineering (lags, rolling stats, calendar features) builds on top of.
select
    train.store_id,
    train.sale_date,
    train.day_of_week,
    train.sales,
    train.customers,
    train.is_open,
    train.is_promo,
    train.state_holiday,
    train.is_school_holiday,
    store.store_type,
    store.assortment,
    store.competition_distance,
    store.has_promo2
from {{ ref('stg_train') }} as train
left join {{ ref('stg_store') }} as store
    on train.store_id = store.store_id
