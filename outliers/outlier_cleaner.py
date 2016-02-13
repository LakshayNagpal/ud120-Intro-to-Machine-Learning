#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error = (net_worths - predictions)**2
    cleaned_data = zip(ages, net_worths, error)

    # for pred,age,networth in zip(predictions, ages, net_worths):
    #     cleaned_data.append((age[0], networth[0], pred[0] - networth[0]))
    #     print "holalala"

    # umar = 0
    # for umar in range(0,len(ages)):
    #     cleaned_data.append(tuple(ages[x], net_worths[x], predictions[x] - net_worths[x]))


    # errors = (net_worths-predictions)**2
    # cleaned_data =zip(ages,net_worths,errors)
    # cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0], reverse=True)
    # limit = int(len(net_worths)*0.1)
        
        
    #     return cleaned_data[limit:]

    #sorting the cleaned data based on the error
    cleaned_data.sort(key=lambda tup: tup[2])

    return cleaned_data[:81]

