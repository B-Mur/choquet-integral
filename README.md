# ChoquetIntegral
The Choquet Integral is a proven fusion strategy. 
Check out any of the publications that we've done on the Choquet Integral (http://derektanderson.com/pubs.html). 

This code is a python class file for learning and using a Choquet Integral. 

## Requirements
Please use a conda environment. I beg you. 

There's probably a way to pip install the requirements.txt, but I'm not that savy yet. Just 
run these commands in your own conda environment, and everything should run smoothly. 

`conda install -yc anaconda cvxopt numpy pandas
 conda install -yc plotly plotly`
 
* _cvxopt_ is the quadratic program package
* _numpy_ come on, you've gotta already know what this is
* _pandas_ I like the dataframes this provides
* _plotly_ used for easy plotting ;)  

##Code
### _*choquet_integral.py*_
#### *Description :* Create a Choquet integral object. Use data to learn the ChI. 
#### *Example :* 
    # create data samples and labels to produce a max aggregation operation
    
    data = np.random.rand(3, 25)
    labels = np.amax(data, 0)
    
    # initialize choquet integral object. 
    chi = ChoquetIntegral()
    
    # train the chi via quadratic program 
    chi.train_chi(data, labels)

    # print out the learned chi variables. (in this case, all 1's) 
    print(chi.fm)
   


`
