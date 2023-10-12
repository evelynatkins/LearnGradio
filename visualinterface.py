
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


def linearRegression(NumberOfValues, NoiseCoeff, m, b):
    # generate random data-set
    #np.random.seed(0) # choose random seed (optional)
    x = np.random.rand(NumberOfValues, 1)
    y = b + m * x + NoiseCoeff*np.random.rand(NumberOfValues, 1)
    
    J = 0 # initialize J, this can be deleted once J is defined in the loop
    w = np.matrix([np.random.rand(),np.random.rand()]) # slope and y-intercept
    a = 0.1 # learning rate step size
    ite = 10 # number of training iterations

    ## Write Linear Regression Code to Solve for w (slope and y-intercept) Here ##
    for p in range (ite):
        for i in range(len(x)):
            # Calculate w and J here
            x_vec = np.matrix([x[i][0],1]) # Setting up a vector for x (x_vec[j] corresponds to w[j])
            h = w*x_vec.T
            w = w-a*(h-y[i])*x_vec
            J = 0.5*(h-y[i])**2
        
        print('Loss:', J)

    ## if done correctly the line should be in line with the data points ##

    print('f = ', w[0,0],'x + ', w[0,1])

    # plot
    plt.scatter(x,y,s=10)
    plt.plot(x, w[0,1] + (w[0,0] * x), linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("linReg.png")
    plt.close()
    
    return "linReg.png"

visualInterface = gr.Interface(
    fn=linearRegression, 
    inputs=[
        gr.Slider(2,1000,label="Number of Values",info="Provide the number of random points you want to use"), 
        gr.Slider(0,5,label="Noise Coefficient",info="Scale the amount of noise in the random dataset"),
        gr.Number(label="Slope",info="Choose a slope for the function determining y values"),
        gr.Number(label="Y-offset", info="Choose a y-offset for the function determining y values")
    ],
    outputs="image", 
    live=True
)

visualInterface.launch()