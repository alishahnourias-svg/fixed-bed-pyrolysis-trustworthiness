# Feature protocols

## A_numeric_only

Composition and operating variables only; no feedstock identity.

**Numeric features:**

- `carbon`
- `hydrogen`
- `nitrogen`
- `oxygen`
- `moisture`
- `volatile_matter`
- `fixed_carbon`
- `ash`
- `particle_size_mm`
- `temperature_c`
- `heating_rate_c_min`
- `gas_flow_rate_l_min`
- `residence_time_min`
- `raw_material_supply_g`

**Categorical features:**

- None

## B_feedstock_identity

Composition and operating variables plus exact feedstock identity.

**Numeric features:**

- `carbon`
- `hydrogen`
- `nitrogen`
- `oxygen`
- `moisture`
- `volatile_matter`
- `fixed_carbon`
- `ash`
- `particle_size_mm`
- `temperature_c`
- `heating_rate_c_min`
- `gas_flow_rate_l_min`
- `residence_time_min`
- `raw_material_supply_g`

**Categorical features:**

- `feedstock`

## C_feedstock_family

Composition and operating variables plus broader feedstock-family taxonomy.

**Numeric features:**

- `carbon`
- `hydrogen`
- `nitrogen`
- `oxygen`
- `moisture`
- `volatile_matter`
- `fixed_carbon`
- `ash`
- `particle_size_mm`
- `temperature_c`
- `heating_rate_c_min`
- `gas_flow_rate_l_min`
- `residence_time_min`
- `raw_material_supply_g`

**Categorical features:**

- `feedstock_family`
