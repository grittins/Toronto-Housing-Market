-- ONE FOR ALL
SELECT pk.year_quarter_key, 
hp._date, hp.area, hp.community, hp.municipality, hp.sales, 
hp.dollar_volume, hp.average_price, hp.new_listings, hp.average_sp_lp, 
hp.average_dom, hp.building_type,
ir.avg_five_year_rates, 
inf.CPI_TRIM, 
rec.canrecdm, rec._indicator
INTO all_info
FROM prim_key AS pk 
    LEFT JOIN home_prices AS hp 
        ON (pk.year_quarter_key = hp.year_quarter_key) 
    LEFT JOIN interest_rate AS ir 
        ON (pk.year_quarter_key = ir.year_quarter_key)
    LEFT JOIN inflation AS inf 
        ON (pk.year_quarter_key = inf.year_quarter_key)
    LEFT JOIN recession_indicator AS rec 
        ON (pk.year_quarter_key = rec.year_quarter_key);

-- Call table
SELECT * 
FROM all_info;

