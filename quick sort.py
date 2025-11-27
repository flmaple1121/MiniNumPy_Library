# Function to partition the array
def partition(array, low, high):
   pivot = array[high] # Choose the last element as pivot
   i = low - 1 # Pointer for smaller elements
   for j in range(low, high):
       if array[j] <= pivot: # If current element is smaller than or equal to pivot
           i += 1
           array[i], array[j] = array[j], array[i] # Swap elements
   # Place the pivot in its correct position
   array[i + 1], array[high] = array[high], array[i + 1]
   return i + 1
# Function to perform Quick Sort
def quick_sort(array, low, high):
   if low < high:
       pi = partition(array, low, high) # Partition index
       quick_sort(array, low, pi - 1) # Recursively sort left subarray
       quick_sort(array, pi + 1, high) # Recursively sort right subarray
# Example usage
data = [8, 7, 2, 1, 0, 9, 6]
print("Unsorted Array:", data)
quick_sort(data, 0, len(data) - 1)
print("Sorted Array:", data)