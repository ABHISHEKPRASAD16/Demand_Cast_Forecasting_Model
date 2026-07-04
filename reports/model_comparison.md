| model               | scope                 |   mape |    rmse |   wape |
|:--------------------|:----------------------|-------:|--------:|-------:|
| LightGBM            | Global (1,115 stores) |   9.02 |  902.21 |   8.75 |
| LSTM                | Global (1,115 stores) |   8.63 |  901.50 |   8.59 |
| Naive               | Store 262             |  15.01 | 4508.36 |  15.73 |
| Seasonal naive (7d) | Store 262             |   9.06 | 2110.39 |   8.64 |
| Moving average (7d) | Store 262             |  15.65 | 4489.00 |  15.98 |
| SARIMAX(2, 1, 1)    | Store 262             |   9.21 | 2223.46 |   8.72 |
| Prophet             | Store 262             |   7.83 | 2046.24 |   8.11 |
| LightGBM            | Store 262             |   5.85 | 1660.39 |   6.12 |
| LSTM                | Store 262             |   6.27 | 1993.25 |   6.62 |