# Team 1 Final Project 

## Team Members
- Anne Lecomte @padawanne - anne-lecomte@hotmail.com
- Derek Mears @mearsdj - derek.j.mears@gmail.com
- Rezwan Ferdous @grittins - grittins1@gmail.com
- Shivali Sahai @shivalisahai - shivali.sahai@gmail.com
- Wayne Jin @jinwei1207 - waynejin0110@gmail.com

## Project Overview: Evolution Toronto House Prices
Our project aims at predicting Toronto house prices and with that; to consider the best timing to buy or sell a property. This question impacts every Torontonian in some way, regardless of their status or current employment situation. We chose to investigate this as we had been wondering how an average person could afford a house in Toronto in the current economy. 
The goal is therefore to predict average house prices while taking into account the type of houses (attached/detached/condo), the location, the timing, interest rates, inflation rate and recession period. This could by used by governments and policy makers looking for ways to balance the growth of house prices, but also by realtors, investors, or any individual interested in the Toronto housing market. 

- **Machine Learning Model:** 
<img width="358" alt="ML Decision Flow Chart" src="https://user-images.githubusercontent.com/104603046/192656877-cbfa1361-aaf9-42f6-a58b-85b0d71beeea.png">

This flowchart indicates how we decided to go with an unstructured ML, using sklearn linear model for price prediction.

- **Database:** We'll be using PostGreSQL for our database.
- **Dashboard:** We're still debating on how to tackle that part of the assignment. We'll most likely be doing some visualizations using Tableau. 

## Resources 
- We are using TREB sold data from the past 21 years.
- sc2.py was used to scrape treb site for pdfs. 
- ML Analasis FlowChart.png
- ML Decision FlowChart.png
- Target Variable Details and Data Structure.xlsx - excel sheet with info on which price series we'll include for forecasting, some data samples from the raw extracts from treb pdf files.
- DataProcessing_ExcelVBA - draft of vba macro to start understanding how excel files are structured and how we might begin to automate processing. only works for midpoint pdf version at the moment.


Linear Regression approach will be modelled for the factors effecting house prices. For the model, the following X-variables will be taken into account. 
- Time Period 
- House-Type 
- Location 
- Interest Rates
- Inflation Rate
- Recession Period

A simple database diagram is sketched for the initial understanding of how the datapoints are connected. A further reiteration is expected. 

![Database_Sketch](https://user-images.githubusercontent.com/104872971/192127719-8e3ef7e1-a358-47a3-b067-08b6667e969c.png)


