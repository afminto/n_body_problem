import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def accel_unit(s_effect, s_cause):
    x = s_cause[0] - s_effect[0]
    y = s_cause[1] - s_effect[1]
    length = sqrt(x**2 + y**2)
    #print(length)
    unit = np.array([x, y]) / length**3
    #print(unit)
    return unit

def accel(particle_i, s_prev_arr, masses):
    acc = np.array([0.,0.])
    for i in range(len(s_prev_arr)):
        if i == particle_i:
            continue
        else:
            #print(type(masses[i]))
            #print(type(acc))
            acc += (6.67e-11 * float(masses[i]) * accel_unit(s_prev_arr[particle_i], s_prev_arr[i]))
    print(acc)
    return acc
   

def gravity_sim(num_particles, masses, s_init, v_init, n_steps, dt):
    s_array = np.zeros((num_particles, n_steps, 2), float)
    v_array = np.zeros((num_particles, n_steps, 2), float)
    s_array[:,0,:] = np.array(s_init, float)
    v_array[:,0,:] = np.array(v_init, float)
    for step in range(0, n_steps-1):
        for particle_i in range(num_particles):
            acc = accel(particle_i, s_array[:,step,:], masses)
            v_array[particle_i,step+1,:] = v_array[particle_i,step,:] + acc * dt
            s_array[particle_i,step+1,:] = s_array[particle_i,step,:] + v_array[particle_i,step,:] * dt
    return s_array, v_array


s_array, v_array = gravity_sim(3, [5.97e27, 1.99e29, 1.0e29], [[1.47e11, 0], [0, 0], [0, 5.0e10]], [[0, 2.9e3], [0, 0], [0,0]], 1000, 10000)

part1_path_x = s_array[0,:,0]
part1_path_y = s_array[0,:,1]
part2_path_x = s_array[1,:,0]
part2_path_y = s_array[1,:,1]
part3_path_x = s_array[2,:,0]
part3_path_y = s_array[2,:,1]
print(v_array)
plt.plot(part1_path_x, part1_path_y, color = 'red')
plt.plot(part2_path_x, part2_path_y, color = 'blue')
plt.plot(part3_path_x, part3_path_y, color = 'green')

plt.show()
plt.close()
