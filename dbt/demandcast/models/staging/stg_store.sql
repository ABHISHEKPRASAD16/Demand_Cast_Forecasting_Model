-- One row per store. Casts + renames only.
select
    cast(store as integer) as store_id,
    storetype as store_type,
    assortment,
    cast(competitiondistance as double) as competition_distance,
    cast(competitionopensincemonth as integer) as competition_open_since_month,
    cast(competitionopensinceyear as integer) as competition_open_since_year,
    cast(promo2 as boolean) as has_promo2,
    cast(promo2sinceweek as integer) as promo2_since_week,
    cast(promo2sinceyear as integer) as promo2_since_year,
    promointerval as promo_interval
from {{ source('raw', 'store') }}
