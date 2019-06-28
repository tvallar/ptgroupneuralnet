Some Physics things:


X,t,Q are parameters p0-p2
p3,p4,p5 same throughout the entire set
F is solution to equation in email



Double_t K(Double_t *x, Double_t *par) {
   return -1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0] + par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));
}


I have those 3 parameters, I am trying to solve for par[3], par[4] and par[5] which remain constant throughout the dataset
F is the solution to the equation above


Keller, D (dmk9m) <dustin@virginia.edu>
	
Attachments11:01 AM (1 hour ago)
	
to me, Yeshwanth
Here is some first trial data to explore.  At this first stage we are just exploring the method and data form.  So if you discover problems please report back.

The general method is this:


1.) Input variable are x_b, t, and Q.  There are 3 parameters P3, P4, and P5 that we want.
2.) Feed inputs into your ML with random weighting and extract the parameters from output
3.) Use the extracted parameters to calculate the observable at the input kinematics and compare to the provided data and find the error (from the difference using least squares)
4.) Use the error to adjust ML and iterate the process


So essentially plug the things in, then get the output and plug those in to the equation and use the error from that equation as the error for the ML model
^ so this will take a custom made machine learning model...


What I am doing:
I am trying to make a machine learning algorithm that will fit multiple equations that share the same parameter.

y1 = (p[0]+p[1] + p[2])*np.cos(x) + p[3]*np.sin(x) + p[0]
y2 = p[1]*p[1]*np.cos(2*x) + p[2] + p[0]*np.sin(5*x)
y3 = (p[3]*p[3] + p[0]*p[0])*np.cos(3*x) + p[2]*np.sin(x) + p[1]

Like the 3 equations above is a good example, more can be found in the pdf


****UPDATED*****
What I am supposed to do:
While fitting is one part of the equation, could aslo do black box learning to calculate the form factors like in some of the papers I've read
I will do this now on the equations I have already set up

Additionally, using SOM to do signal / background classification, then using SOM for regression.


On top of that, use the models I create for determining the form factor to be able to get a continuous model



The algorithm needs to be fairly versatile such that it can be used on any equations of this nature
The equation I will be fitting at the end has 5 dimension, but 4 are set and cos(phi) is the x-axis make up the datapoints I will be fitting to
I can use as many datapoints as possible, but according to dustin they cost "half a million each" so probably best to try and minimize the number needed


Where to start:
Simultaneous fitting with shared parameters
the Minuit Cern fitting function, Chi-squared minimization


Things I am waiting for:
A simplified version of the QM equations
An exercise sheet that might give me


https://www.mathworks.com/matlabcentral/answers/47389-simultaneously-fit-multiple-data-sets-with-one-of-the-fit-parameters-common-between-all-data-sets
^ This is very helpful


Some Results:
When there isn't a constant addition by one of the parameters, they are far less accurate.


Look up Chi Squared more, find iterative fitting routine with Chi Squared
