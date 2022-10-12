-- Creating tables for Housing_Market_DB
CREATE TABLE prim_key (
	Year_Quarter_Key INT NOT NULL,
	_Year INT NOT NULL,
    Quarter INT NOT NULL,
	PRIMARY KEY (Year_Quarter_Key)
);

CREATE TABLE home_prices (
    Year_Quarter_Key INT NOT NULL,
	_Year INT NOT NULL,
    Quarter INT NOT NULL,
	_no INT NOT NULL,
	Area VARCHAR(7) NOT NULL, 
    Municipality VARCHAR(25) NOT NULL, 
    Community VARCHAR(44) NOT NULL,
    Sales INT NOT NULL, 
    Dollar_Volume INT NOT NULL, 
    Average_Price FLOAT NOT NULL, 
    New_Listings INT NOT NULL, 
    Average_SP_LP FLOAT, 
    Average_DOM FLOAT NOT NULL,
    _date VARCHAR(6) NOT NULL, 
    Building_Type VARCHAR(16) NOT NULL,
	FOREIGN KEY (Year_Quarter_Key) REFERENCES prim_key (Year_Quarter_Key)
);

CREATE TABLE interest_rate (
    Year_Quarter_Key INT NOT NULL,
    avg_five_year_rates FLOAT NOT NULL,
	FOREIGN KEY (Year_Quarter_Key) REFERENCES prim_key (Year_Quarter_Key),
    PRIMARY KEY (Year_Quarter_Key)
);

CREATE TABLE recession_indicator (
    Year_Quarter_Key INT NOT NULL,
	_Year INT NOT NULL,
    Quarter INT NOT NULL,
	_date DATE NOT NULL, 
    CANRECDM FLOAT NOT NULL,
    _Indicator BOOLEAN NOT NULL,
	FOREIGN KEY (Year_Quarter_Key) REFERENCES prim_key (Year_Quarter_Key),
    PRIMARY KEY (Year_Quarter_Key)
);