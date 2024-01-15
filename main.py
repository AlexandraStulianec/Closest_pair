import math
import random
import csv
import time
import tracemalloc
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import bridson  # Import the bridson library for Poisson Disk Sampling
from sklearn.cluster import KMeans # Import sklearn to have the KMeans function


def euclidean_distance(p1, p2):

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def brute_force_closest_pair(points):
    # Base case: Brute force method for finding closest pair when |P| <= 3
    min_distance = float('inf')
    closest_pair = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclidean_distance(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])

    return min_distance, closest_pair


def split_y(Y, PL):
    YL = []
    YR = []

    #tuple_PL = [tuple(point) for point in PL]

    for point in Y:
        if point in PL:
            YL.append(point)
        else:
            YR.append(point)

    return YL, YR


def closest_pair_recursive(X, Y):
    n = len(X)

    # Base case: Brute force when |P| <= 3
    if n <= 3:
        return brute_force_closest_pair(X)

    # Divide phase
    mid = n // 2
    XL = X[:mid]
    XR = X[mid:]
    YL, YR = split_y(Y, XL)

    # Recursively find closest pair in left and right subsets
    deltaL, pairL = closest_pair_recursive(XL, YL)
    deltaR, pairR = closest_pair_recursive(XR, YR)

    # Determine minimum delta from left and right subsets
    delta = min(deltaL, deltaR)

    # Combine phase - strip/band case
    Y_prime = [point for point in Y if abs(point[0] - X[mid][0]) < delta]

    strip_size = len(Y_prime)

    # Check for closest pair in the strip
    min_strip_distance = delta
    min_strip_pair = None

    for i in range(strip_size):
        for j in range(i + 1, min(i + 7, strip_size)):
            distance = euclidean_distance(Y_prime[i], Y_prime[j])
            if distance < min_strip_distance:
                min_strip_distance = distance
                min_strip_pair = Y_prime[i], Y_prime[j]

    # Compare strip pair with pairs from left and right subsets
    if min_strip_distance < delta:
        return min_strip_distance, min_strip_pair
    elif deltaL < deltaR:
        return deltaL, pairL
    else:
        return deltaR, pairR


def closest_pair(points):
    # Sort points by x-coordinate
    sorted_points_x = sorted(points, key=lambda x: x[0])

    # Sort points by y-coordinate
    sorted_points_y = sorted(points, key=lambda x: x[1])

    result_distance, result_pair = closest_pair_recursive(sorted_points_x, sorted_points_y)

    return result_distance, result_pair

def measure_time_and_memory(func, arr):
    start_time = time.perf_counter_ns()
    tracemalloc.start()

    # Execute the function
    func(arr)

    end_time = time.perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1000000000

    # Measure memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = peak / 1024  # Convert to kilobytes

    return  elapsed_time,memory_usage


def generate_poisson_disk_samples(width, height, r, k, set_size):
    points = bridson.poisson_disc_samples(width, height, r, k)
    return [(int(x), int(y)) for x, y in points][:set_size]

def kmeans_clustering(points, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(points)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return labels, cluster_centers

def generate_grid_points(grid_size, set_size):
    scale = max(1, round((grid_size ** 2 / set_size) ** 0.5))
    points = [(x * scale, y * scale) for x in range(grid_size) for y in range(grid_size)]
    random.shuffle(points)
    return points[:set_size]


def generate_random_points(num_points, x_range=(-1000, 1000), y_range=(-1000, 1000)):
    points = []

    while len(points) < num_points:
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        points.append((x, y))

    return points


# User inputs
print("Write a file name for the computation for the time&space values (ex: file.csv):")
f_name1 = input()

print("Write a file name for the points (ex: file.csv):")
f_name2 = input()

print("Number of sets you want to generate:")
run = int(input())

print("Set size:")
set_size = int(input())

# File setup
f = open(f_name1, 'w', newline='')
f2 = open(f_name2, 'w', newline='')
writer = csv.writer(f)
writer2 = csv.writer(f2)

header = ["Set", "ClosestPair", "ClosestPairDistance", "Time", "Space"]
writer.writerow(header)

header2 = ["Points"]
writer2.writerow(header2)

times = []
output_line = []
output_line2 = []

while run:
    # Generate points using Poisson Disk Sampling , clustered points, in a grid pattern and randomized

    # points = generate_poisson_disk_samples(1000, 1000, 10, 30, set_size)
    points = generate_grid_points(50, set_size) #grid_size: if set_size < 10000 - 50 else 100
    # points = generate_random_points(set_size)

    #num_clusters = 5 # Set the number of clusters as needed - I increased the size of the num_clusters to be set_size/10
    #labels, cluster_centers = kmeans_clustering(points, num_clusters) - have this also when using this case to find the centers


    # Measure time and memory

    elapsed_time, memory_usage = measure_time_and_memory(closest_pair, points)
    #elapsed_time, memory_usage = measure_time_and_memory(brute_force_closest_pair, points)

    result_distance, result_pair = closest_pair(points)
    formatted_pair = f"({result_pair[0]}, {result_pair[1]})"
    print(formatted_pair)


    output_line = [set_size, formatted_pair, result_distance, elapsed_time, memory_usage]
    output_line2 = [points]
    writer.writerow(output_line)
    writer2.writerow(output_line2)

    # User interface for showing the closest pair (and the cluster centers for the clustered case)
    #plt.scatter(*zip(*cluster_centers), marker='X', color='black', label='Cluster Centers') - this for the clustered case
    plt.scatter(*zip(*points), label=f"Set {set_size}")

    # The closest pair is marked with a line
    closest_pair_line_x = [result_pair[0][0], result_pair[1][0]]
    closest_pair_line_y = [result_pair[0][1], result_pair[1][1]]
    plt.plot(closest_pair_line_x, closest_pair_line_y, color='red', linestyle='--', label='Closest Pair')

    run -= 1

# Show the plot
plt.legend()
plt.title('Sample')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()

f.close()
f2.close()
