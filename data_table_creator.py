import datetime
import pandas as pd
import pandas_market_calendars as pmc
current = datetime.datetime.now().date()
def create_date_table(start='2000-01-01', end=current):

    hcal = pmc.get_calendar('TSX')
    hcalus = pmc.get_calendar('NYSE')
    cb = pd.offsets.CBMonthEnd(calendar=hcal, holidays=hcal.holidays().holidays)
    cd = pd.offsets.CustomBusinessDay(calendar=hcal, holidays=hcal.holidays().holidays)
    cbus = pd.offsets.CBMonthEnd(calendar=hcalus, holidays=hcalus.holidays().holidays)
    cdus = pd.offsets.CustomBusinessDay(calendar=hcalus, holidays=hcalus.holidays().holidays)
    cdpd = pd.offsets.CustomBusinessDay(n=-1,calendar=hcal, holidays=hcal.holidays().holidays)
    cduspd = pd.offsets.CustomBusinessDay(n=-1, calendar=hcalus, holidays=hcalus.holidays().holidays)
    daymap={'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Friday':6,'Saturday':7}
    df = pd.DataFrame({"Full_Date": pd.date_range(start, end)})
    df["Date_Key"]=df.Full_Date.dt.strftime('%Y%m%d')
    df["Date_Name"]=df.Full_Date.dt.strftime('%b %#d, %Y')
    df["Day_of_Week"]=df.Full_Date.dt.dayofweek
    df["Day_Name_of_Week"]=df.Full_Date.dt.day_name()
    df['Day_of_Week'] = df.apply(lambda x: daymap[x.Day_Name_of_Week], axis=1)
    df["Day_of_Month"]=df.Full_Date.dt.day
    df["Day_of_Year"]=df.Full_Date.dt.dayofyear
    df["Weekday_Weekend"] = df['Day_of_Week'].apply(lambda x: "Weekend" if x >=5 else "Weekday")
    df["Week_of_Year"]=df.Full_Date.dt.isocalendar().week
    df["Month_Name"]=df.Full_Date.dt.strftime('%B')
    df["Is_Last_Day_of_Month"]=df.Full_Date.dt.is_month_end
    df["Calendar_Quarter"]=df.Full_Date.dt.quarter
    df["Calendar_Year"]=df.Full_Date.dt.year
    df["Calendar_Month"]=df.Full_Date.dt.month
    df["Calendar_Year_Month"]=df.Full_Date.dt.strftime('%Y-%#m')
    df['Calendar_Year_Quarter']=df.Full_Date.dt.to_period('Q').dt.strftime('%Y%q')
    df["Is_Canadian_Holiday"]=df.Full_Date.isin(cb.holidays)
    df["Is_US_Holiday"]=df.Full_Date.isin(cbus.holidays)
    df["Previous_Business_Date_Cdn"]=(df.Full_Date+cdpd).dt.strftime('%Y%m%d')
    df["Previous_Business_Date_US"]=(df.Full_Date+cduspd).dt.strftime('%Y%m%d')
    df["Insert_Audit_Key"]=-2
    df["Update_Audit_Key"]=-2
    df["Load_Date_Time"]=datetime.datetime.now()
    df["Is_Last_Business_Day_of_Month_Cdn"] = df['Full_Date'].apply(lambda x: cb.is_on_offset(x))
    df["Is_Last_Business_Day_of_Quarter_Cdn"] = df['Full_Date'].apply(lambda x: (cb.is_on_offset(x)&(x.month in [3,6,9,12])))
    return df

if __name__ == '__main__':
    create_date_table()

