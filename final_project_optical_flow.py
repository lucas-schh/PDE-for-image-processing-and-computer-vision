#Final Project - Lucas Schwing
# MPI-SINTEL DATA SET
# Copyright (c) 2012 Daniel Butler, Jonas Wulff, Garrett Stanley, Michael Black, 
# Max-Planck Institute for Intelligent Systems, Tuebingen

import os
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# import_images_from_directory and plot_uv are used to import the images
# and transform it into matrices and to plot u and v during each iterations
# These functions have been made with the help of internet and ChatGPT

def import_images_from_directory(direc, scale):
    list_images_RGB = [];
    list_images_gray = [];
    
    for image_file in os.listdir(direc):
        image_path = os.path.join(direc, image_file)
        img_RGB = Image.open(image_path)
        
        # Rescale the images
        width, height = img_RGB.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_RGB_resized = img_RGB.resize((new_width, new_height))

        img_array_RGB = np.array(img_RGB_resized, dtype=np.float_)
        #each frame represent a different time (frame 0 = I(x,y,0))
        list_images_RGB.append(img_array_RGB)
        
        #Converting to gray scale images to try the optical flow from class
        img_gray = img_RGB_resized.convert('L')
        img_array_gray = np.array(img_gray, dtype=np.float_)
        list_images_gray.append(img_array_gray)
        
    I = np.array(list_images_RGB) #I(x,y,t)
    I_gray = np.array(list_images_gray)
    return I, I_gray

# Draw arrows which represent the optical flow on the image
def draw_arrows(image, u, v):
    for x in range(0,image.shape[1],2):
        for y in range(0,image.shape[0],2):
            plt.arrow(x, y, u[y][x]*2, v[y][x]*2, color='blue')
    return

def compute_partial_d(I, arg, delta):
    if(arg=="x"):
        if(delta>I.shape[2]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        I_tmp = np.copy(I)
        for t in range(len(I)):
            for y in range(I.shape[1]):
                for x in range(0,delta):
                    I_tmp[t][y][x] = (I[t][y][x+delta]-I[t][y][x])/delta #Forward diff for the first element
                for x in range(delta, I.shape[2]-1-delta):
                    I_tmp[t][y][x] = (I[t][y][x+delta]-I[t][y][x-delta])/(2*delta) #Central diff
                for x in range(I.shape[2]-1-delta, I.shape[2]):
                    I_tmp[t][y][x] = (I[t][y][x]-I[t][y][x-delta])/delta #backward diff for the last element
    if(arg=="y"):
        if(delta>I.shape[1]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        I_tmp = np.copy(I)
        for t in range(len(I)):
            for y in range(0,delta):
                I_tmp[t][y] = (I[t][y+delta]-I[t][y])/delta
            for y in range(delta, I.shape[1]-1-delta):
                I_tmp[t][y] = (I[t][y+delta]-I[t][y-delta])/(2*delta)
            for y in range(I.shape[1]-1-delta, I.shape[1]):
                I_tmp[t][y] = (I[t][y]-I[t][y-delta])/delta
    if(arg=="t"):
        if(delta>I.shape[0]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        I_tmp = np.copy(I)
        for t in range(0, delta):
            I_tmp[t] = (I[t+delta]-I[t])/delta
        for t in range(delta, len(I)-1-delta):
            I_tmp[t] = (I[t+delta]-I[t-delta])/(2*delta)
        for t in range(len(I)-1-delta, len(I)):
            I_tmp[t] = (I[t]-I[t-delta])/delta
    return I_tmp
   

#Compute partial derivative of u or v
def compute_partial_d_UV(func, arg, delta):
    func_tmp = np.copy(func)
    if(arg=="x"):
        if(delta>func.shape[1]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        for y in range(func.shape[0]):
            for x in range(0,delta):
                func_tmp[y][x] = (func[y][x+delta]-func[y][x])/delta
            for x in range(delta, func.shape[1]-1-delta):
                func_tmp[y][x] = (func[y][x+delta]-func[y][x-delta])/(2*delta)
            for x in range(func.shape[1]-1-delta, func.shape[1]):
                func_tmp[y][x] = (func[y][x]-func[y][x-delta])/delta
    if(arg=="y"):
        if(delta>func.shape[0]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        for y in range(0, delta):
            func_tmp[y] = (func[y+delta]-func[y])/delta
        for y in range(delta, func.shape[0]-1-delta):
            func_tmp[y] = (func[y+delta]-func[y-delta])/(2*delta)
        for y in range(func.shape[0]-1-delta, func.shape[0]):
            func_tmp[y] = (func[y]-func[y-delta])/delta
    return func_tmp

#Compute second partial derivative of u or v
def compute_second_partial_d_UV(func, arg, delta):
    func_tmp = np.copy(func)
    if(arg=="x"):
        if(delta>func.shape[1]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        for y in range(func.shape[0]):
            for x in range(0,delta):
                func_tmp[y][x] = (func[y][x+2*delta]-2*func[y][x+delta]+func[y][x])/delta**2 #Forward for the first element
            for x in range(delta, func.shape[1]-1-delta):
                func_tmp[y][x] = (func[y][x+delta]-2*func[y][x] + func[y][x-delta])/delta**2 #Central
            for x in range(func.shape[1]-1-delta, func.shape[1]):
                func_tmp[y][x] = (func[y][x]-2*func[y][x-delta]+func[y][x-2*delta])/delta**2 #Backward
    if(arg=="y"):
        if(delta>func.shape[0]/2): #for the central difference
            print("ERROR: delta is too large")
            return
        for y in range(0, delta):
            func_tmp[y] = (func[y+2*delta]-2*func[y+delta]+func[y])/delta**2
        for y in range(delta, func.shape[0]-1-delta):
            func_tmp[y] = (func[y+delta]-2*func[y] + func[y-delta])/delta**2
        for y in range(func.shape[0]-1-delta, func.shape[0]):
            func_tmp[y] = (func[y]-2*func[y-delta]+func[y-2*delta])/delta**2
    return func_tmp

# Gradient Descent from the course
def gradient_descent_gray(tau, I_x, I_y, I_t, u_start, v_start):
    u=u_start
    v=v_start
    
    u_values = [] # Store u values for each iteration
    v_values = [] # Store v values for each iteration

    # Compute PDE
    for i in range(tau):
        new_u=np.zeros((I.shape[1], I.shape[2]))
        new_v=np.zeros((I.shape[1], I.shape[2]))
        u_xx=compute_second_partial_d_UV(u, "x", delta)
        u_yy=compute_second_partial_d_UV(u, "y", delta)
        v_xx=compute_second_partial_d_UV(v, "x", delta)
        v_yy=compute_second_partial_d_UV(v, "y", delta)

        for y in range(u.shape[0]):
            for x in range(u.shape[1]):
                new_u[y][x]=u[y][x]+delta_tau*(-l*(I_x[y][x]*u[y][x]+I_y[y][x]*v[y][x]+I_t[y][x])*I_x[y][x]+(1-l)*(u_xx[y][x]+u_yy[y][x]))
                new_v[y][x]=v[y][x]+delta_tau*(-l*(I_x[y][x]*u[y][x]+I_y[y][x]*v[y][x]+I_t[y][x])*I_y[y][x]+(1-l)*(v_xx[y][x]+v_yy[y][x]))
                
        u=new_u
        v=new_v
        
        u_values.append(np.mean(u))
        v_values.append(np.mean(v))
    return u, v, u_values, v_values

# Gradient Descent using RGB images - Final Project
def gradient_descent_RGB(tau, I_x, I_y, I_t, u_start, v_start):
    u=u_start # Start from the last value of the previous frame
    v=v_start
    
    u_values = [] # Store u values for each iteration
    v_values = [] # Store v values for each iteration

    # Compute PDE
    for i in range(tau):
        new_u=np.zeros((I.shape[1], I.shape[2]))
        new_v=np.zeros((I.shape[1], I.shape[2]))
        
        u_x=compute_partial_d_UV(u, "x", delta)
        u_y=compute_partial_d_UV(u, "y", delta)
        u_xy=compute_partial_d_UV(compute_partial_d_UV(u, "x", delta), "y", delta)
        u_xx=compute_second_partial_d_UV(u, "x", delta)
        u_yy=compute_second_partial_d_UV(u, "y", delta)
        v_x=compute_partial_d_UV(v, "x", delta)
        v_y=compute_partial_d_UV(v, "y", delta)
        v_xy=compute_partial_d_UV(compute_partial_d_UV(v, "x", delta), "y", delta)
        v_xx=compute_second_partial_d_UV(v, "x", delta)
        v_yy=compute_second_partial_d_UV(v, "y", delta)

        for y in range(u.shape[0]):
            for x in range(u.shape[1]):
                R_x=I_x[y][x][0]
                R_y=I_y[y][x][0]
                R_t=I_t[y][x][0]
                G_x=I_x[y][x][1]
                G_y=I_y[y][x][1]
                G_t=I_t[y][x][1]
                B_x=I_x[y][x][2]
                B_y=I_y[y][x][2]
                B_t=I_t[y][x][2]
                eps=10**-5 # Adding epsilon in denominator to avoid dividing by 0
                u_fidelity_term=2*((R_x*u[y][x]+R_y*v[y][x]+R_t)*R_x+(G_x*u[y][x]+G_y*v[y][x]+G_t)*G_x+(B_x*u[y][x]+B_y*v[y][x]+B_t)*B_x)
                v_fidelity_term=2*((R_x*u[y][x]+R_y*v[y][x]+R_t)*R_y+(G_x*u[y][x]+G_y*v[y][x]+G_t)*G_y+(B_x*u[y][x]+B_y*v[y][x]+B_t)*B_y)
                u_regularization_term=(u_xx[y][x]*(u_y[y][x]**2)-2*u_x[y][x]*u_y[y][x]*u_xy[y][x]+u_yy[y][x]*(u_x[y][x]**2)+eps*(u_xx[y][x]+u_yy[y][x]))/math.pow(math.sqrt(u_x[y][x]**2+u_y[y][x]**2+eps),3)
                v_regularization_term=(v_xx[y][x]*(v_y[y][x]**2)-2*v_x[y][x]*v_y[y][x]*v_xy[y][x]+v_yy[y][x]*(v_x[y][x]**2)+eps*(v_xx[y][x]+v_yy[y][x]))/math.pow(math.sqrt(v_x[y][x]**2+v_y[y][x]**2+eps),3)
                # u_regularization_term=2*(u_xx[y][x]+u_yy[y][x]) # This is to try with a penalize term squared
                # v_regularization_term=2*(v_xx[y][x]+v_yy[y][x])
                new_u[y][x]=u[y][x]+delta_tau*(-l*u_fidelity_term+(1-l)*u_regularization_term)
                new_v[y][x]=v[y][x]+delta_tau*(-l*v_fidelity_term+(1-l)*v_regularization_term)
                
        u=new_u
        v=new_v
        
        u_values.append(np.mean(u))
        v_values.append(np.mean(v))
    return u, v, u_values, v_values

def plot_uv(u_values, v_values):
    plt.plot(u_values, label='u')
    plt.plot(v_values, label='v')
    plt.xlabel('Iteration (tau)')
    plt.ylabel('Value')
    plt.title('Values of u and v over iterations')
    plt.legend()
    plt.show()

I, I_gray = import_images_from_directory('./MPI_Sintel-training_images/training/albedo/alley_1/', 0.2)

#I[t][y][x]

#Parameters
tau = 100 #Number of iterations of computing the PDE
l = 0.3 #Lambda
delta = 1 #For derivative
delta_tau = 5*10**-6

# Computing for gray images
# I_t = compute_partial_d(I_gray, "t", delta)
# I_x = compute_partial_d(I_gray, "x", delta)
# I_y = compute_partial_d(I_gray, "y", delta)

# Computing for RGB images
I_t = compute_partial_d(I, "t", delta)
I_x = compute_partial_d(I, "x", delta)
I_y = compute_partial_d(I, "y", delta)

u=np.zeros((I.shape[1], I.shape[2])) #u[y][x]
v=np.zeros((I.shape[1], I.shape[2])) #v[y][x]

for t in range(len(I)): 
    # Gradient Descent for gray images
    # u, v, u_values, v_values = gradient_descent_gray(tau, I_x[t], I_y[t], I_t[t], u, v)
    # draw_arrows(I_gray[t], u, v)
    # plt.imshow(I_gray[t], cmap="gray")
    
    #Gradient Descent for RGB images
    u, v, u_values, v_values = gradient_descent_RGB(tau, I_x[t], I_y[t], I_t[t], u, v)
    draw_arrows(I[t], u, v)
    plt.imshow(I[t].astype(np.int_))
    
    plt.show()
    plot_uv(u_values, v_values)