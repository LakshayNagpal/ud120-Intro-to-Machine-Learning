#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count = 0
count1 = 0

# for user in enron_data:
# 	if(enron_data[user]["poi"] == 1):
# 		count+=1

# print count


#print enron_data.keys()
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]


for user in enron_data:
	if(enron_data[user]["salary"]!= "NaN"):
		count+=1
	if(enron_data[user]["email_address"]!= "NaN"):
		count1+=1

print count
		
print count1

count = 0.0
for user in enron_data:
	if(enron_data[user]["total_payments"] == "NaN" and enron_data[user]["poi"] == True ):
		count+=1

print count

print count/len(enron_data)*100

count = 0.0
for user in enron_data:
	if(enron_data[user]["total_payments"] == "NaN"):
		count+=1

print count

count = 0.0
for user in enron_data:
	if(enron_data[user]["poi"] == True):
		count+=1

print count

print enron_data[enron_data.keys()[0]].values()