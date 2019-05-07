# CIS-242-HW7-Final-Project

# TOKOPANDAI PROJECT

# Abdullah Alsayari,
# Matthew Taruno,
# Akira Ranjan Sah

# Project Overview:
For this project, we chose to apply the machine learning techniques learned in class to data from a startup called Toko Pandai, based in Indonesia, which literally means “Smart Store.”

57.4% of Indonesia’s GDP is from the private consumption market. In this private consumption market, there are two types of stores: traditional stores (Indonesia has 3.4 million traditional trade stores) and opposed to corporation-owned trade stores (Indonesia has 36000 corporate-owned stores). Traditional stores are stored owned by typically poorer individuals who live in smaller villages, they are not corporate owned. So these traditional stores are the stores that TokoPandai aims to give power towards and help, so that they may be able to fight and compete against the more systematic, dominant presence of modern trade stores.

To give traditional stores competitive power, TokoPandai creates a phone application that aims to seamlessly connect the principal (like Unilever), the distributor (Unilever owns around 600-700 distributors), and the store (each distributor holds several thousand stores). All three parties benefit from this connected application platform that TokoPandai creates, but for this project, to narrow the scope (it would simply take too long to explain the perspectives of all three parties), the focus our machine learning problem aims to solve will just be to solve a problem from the distributor’s point of view. 

What is this aforementioned problem for this distributor that we aim to solve? Traditional stores need to make invoice payments to distributors. An invoice payment from one store, is a transaction made from that particular store to a distributor that covers a time period’s worth (usually a month) of items. This item list, which is charged by an invoice, include a variety of many products, including a variety of soap products, chips, toothbrushes, ice cream, and many other products that a store might need to get from a distributor. This whole list is charged on one invoice. For this project, we predict the difference between the total amount of sales and the balance amount, or in other words the difference between what the store pays at a given time period and how much it is supposed to pay. If the difference is anything other than zero, that means the store is paying it’s invoice partially as opposed to a full payment. This is useful for the distributor to know because based on our input features we input in our UI, the distributor will know how likely it is for a store to give them partial payments or full payments. The more information a distributor knows upfront, the more it is able to prepare for its inventory management and resource allocation. Since we also used outlet code as an input, it can query particular tendencies of a particular store’s tendencies to pay early or late as well using our UI.

The data we have was initially given in the form of an SQL database. Then from this large SQL database (which was quite troublesome to navigate and execute the script because it had multiple tables). Which leads to our data processing and exploration stage, which for a this final project, is pivotal to both the accuracy of our predictions, and the motivating ideas.

# Data Processing:
Our data included many tables: the respective names include distributors, users (shop owners), invoices, and transactions. Each of the tables we had included many features including amount, prices, outlet and distributors’ id, and quantity, with a total of like. Into our model since using that many features can cause the model to overfit. Therefore, we decided on selecting reasonable number of features, and ones that would include many and relevant data to our the output value. 

The first step of this process was getting the data cleaned, this was probably the hardest and most time consuming part of the project. We had a lot of columns to look at, and before we can decide on which columns we wanted as (X) and which one we wanted as our (y), we needed to check if the columns or input features that we chose had a sufficient amount of clean data to back it up. We started this process of data cleaning by writing down every column in each of the tables we had. Each column that was empty or nearly empty was removed from our column list. Next, we had to decide on our (y) column to know which features were relevant to our model. At first we wanted to predict (y) as the probability of risk for banks to loan store owners money. The problem with this output value was that we lacked in demographic data and also training data giving of this actual (y) value - in other words, we have no previous historical data for how stores are loaned money because the startup only plans to roll out the loaning in the near future. 

Resultatively, after careful consideration of all features, we decided that the best thing to predict would be the difference between cash memo balance amount and cash memo total amount of product that a store cells based on the following features: (Outlet_code, Cash_memo_balance_amount, product quantity, DPD, and Product Price) We felt that these were a good amount of features that would result in a good accurate model. Also, to solve our problem with calculating DPD, we ended up combining data from two different tables - from excel_dump and transactions to get the DPD feature column. Also for the quantity column, we also dropped all the values where the quantity was negative. The data is sent straight from Unilever, so the data points that made not much accurate sense, which could harm the data, were dropped. The following screenshot gives an example of the product queries where quantities was initially negative.

 
All rows with negative quantities, as shown here, were dropped.

Even though now we had (X) features that are related to our (y), there were some issues with our features. For example, two of our columns included “Invoice_due_date” and “invoice_sales_date” . These two features were needed to generate Day Past Due (DPD). DPD is calculated by subtracting the sales data from the due date, and to do that, we needed to use an existing python library to be able to perform mathematical operations on date. We also had to deal with date that come in different formats, for which we also used an existing python library. 
Model Design and Test: 

With models, we discussed several possibilities to use with our data. The problem was that the output data (y) was a numerical data, so we couldn’t use something like general classification model or decision tree. We wanted to start with a simple and somewhat naive model that will test our data and see if we have any problems with our data, or if there was something missing in it. 

The first model that we used was a simple linear regression (SLR). A little overview about linear regression, it uses the idea of a line equation*:  

With an actual linear regression, the equation would be somewhat different*: 


The (Xi) here are the different features that we get from our data. The (Bi) are the weight or coefficients that are used to minimize the error in the output value, or more accurately, the distance from the actual (y) value if it is represented in a graph. 

With Python, we had two libraries to implement LR, and we ended up using Scikit Learn, which was fairly simple to implement. With Scikit Learn, we only needed to turn both the (X) and (y) to data frames and then we would take the training (X) and (y) sets and use them as parameters for the fit function. We then used the prediction method with test (X) as parameter to get the predicted (y) and compare it with the test (y) set. We also used the Normalized Root-mean-square deviation (NRMSD) to find the error percentage. We used this equation to calculate NRMSD**:

We started with 10000 rows to test, and then 100000 rows. The error percentage improved, according to our findings, from 3% to 1% error.

We then moved to using more robust model, Random Forest. One reason for choosing such a model was the fact that we can fine-tune the hyperparameters of Random Forest, such as number of decision trees generated and max depth of the tree. For implementing Random Forest, we used Scikit Learn library. In order to create an accurate model, we tested the model with different hyperparameters and at the same time, we tried to be mindful of the running time so that we don’t end up with exponential running time. For example, one of the hyperparameters of Random Forest is the number of iterations of the model takes three seconds for each iteration, so we had to use a relatively small number to run, but at the same time, big enough to create an accurate model. 

The categorical data we had was another reason as to why we chose Random Forest. It is also a good model in the sense that it doesn’t require normalization done on the data before putting it in the model. 

We have also tried using linear regression algorithm to create our model. However, after looking at our plots, it seemed that the relationship between the the features and the output were not linear, so that is another reason why we preferred random forest.

# UI Design: 
For UI design of the project, we mainly used ipywidgets package and imported several tools from it that helped us make the final representation and looks for the project. We used different tools such as Text fields, Sliders, and box layout to allow labels to be set next to the widget element itself. We used several Text fields to take most of the features as inputs, and we used one slider to handle the DPD (Day Past Due) feature. 

We added a button that, upon clicking, the values in the text field and slider widgets will be placed for input variables. The input variables are then passed down as parameters and are included as part of the (X) features. The model will use the features to generate the (y) value, difference, which will be presented on the difference label as the result of the prediction value.
