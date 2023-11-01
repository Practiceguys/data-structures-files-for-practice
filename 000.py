import time


def merge_sortt(arr):
    # Merge Sort
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursive calls
        left_half = merge_sortt(left_half)
        right_half = merge_sortt(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr
# Define data structure classes

class Array:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []
        self.type = None
    def insert(self, item):
        if (len(self.items))<self.max_size:
            self.items.append(item)
        else:
            raise OverflowError("array is full")

    def delete(self, item):
        if item in self.items:
            self.items.remove(item)
        else:
            print("Item not found in the array.")

   
    def sort(self, algorithm):
        if algorithm == 1:
            self.bubble_sort()
        elif algorithm == 2:
            self.selection_sort()
        elif algorithm == 3:
            self.insertion_sort()
        elif algorithm == 4:
            self.merge_sort()
        elif algorithm == 5:
            self.quick_sort()
        elif algorithm == 6:
            self.heap_sort()
        elif algorithm == 7:
            self.radix_sort()
        elif algorithm == 8:
            self.counting_sort()
        else:
            print("Invalid input!\n")

    def bubble_sort(self):
        # Bubble Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.items[j] > self.items[j + 1]:
                    self.items[j], self.items[j + 1] = self.items[j + 1], self.items[j]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Bubble Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def selection_sort(self):
        # Selection Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.items[j] < self.items[min_idx]:
                    min_idx = j
            self.items[i], self.items[min_idx] = self.items[min_idx], self.items[i]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Selection Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def insertion_sort(self):
        # Insertion Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(1, n):
            key = self.items[i]
            j = i - 1
            while j >= 0 and key < self.items[j]:
                self.items[j + 1] = self.items[j]
                j -= 1
            self.items[j + 1] = key
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Insertion Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def merge_sort(self):
        # Merge Sort
        start_time = time.time()
        if len(self.items) > 1:
            mid = len(self.items) // 2
            left_half = self.items[:mid]
            right_half = self.items[mid:]

            # Recursive calls
            left_half = merge_sortt(left_half)
            right_half = merge_sortt(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    self.items[k] = left_half[i]
                    i += 1
                else:
                    self.items[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                self.items[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                self.items[k] = right_half[j]
                j += 1
                k += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Merge Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort(self):
        # Quick Sort
        start_time = time.time()
        self.quick_sort_recursive(0, len(self.items) - 1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Quick Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort_recursive(self, low, high):
        if low < high:
            pivot = self.partition(low, high)
            self.quick_sort_recursive(low, pivot - 1)
            self.quick_sort_recursive(pivot + 1, high)

    def partition(self, low, high):
        pivot = self.items[high]
        i = low - 1
        for j in range(low, high):
            if self.items[j] < pivot:
                i += 1
                self.items[i], self.items[j] = self.items[j], self.items[i]
        self.items[i + 1], self.items[high] = self.items[high], self.items[i + 1]
        return i + 1
    def heap_sort(self):
        # Heap Sort
        start_time = time.time()
        n = len(self.items)

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i)

        for i in range(n - 1, 0, -1):
            self.items[i], self.items[0] = self.items[0], self.items[i]
            self.heapify(i, 0)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Heap Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def heapify(self, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.items[left] > self.items[largest]:
            largest = left

        if right < n and self.items[right] > self.items[largest]:
            largest = right

        if largest != i:
            self.items[i], self.items[largest] = self.items[largest], self.items[i]
            self.heapify(n, largest)

    def radix_sort(self):
        # Radix Sort
        start_time = time.time()
        max_num = max(self.items)
        exp = 1

        while max_num // exp > 0:
            self.counting_sort_radix(exp)
            exp *= 10

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Radix Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def counting_sort_radix(self, exp):
        n = len(self.items)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (self.items[i] // exp)
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (self.items[i] // exp)
            output[count[index % 10] - 1] = self.items[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(n):
            self.items[i] = output[i]

    def counting_sort(self):
        # Counting Sort
        start_time = time.time()
        max_num = max(self.items)
        min_num = min(self.items)
        range_of_elements = max_num - min_num + 1

        count = [0] * range_of_elements
        output = [0] * len(self.items)

        for i in range(len(self.items)):
            count[self.items[i] - min_num] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]

        i = len(self.items) - 1
        while i >= 0:
            output[count[self.items[i] - min_num] - 1] = self.items[i]
            count[self.items[i] - min_num] -= 1
            i -= 1

        for i in range(len(self.items)):
            self.items[i] = output[i]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Counting Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")
        
    def search(self, target, algorithm):
        if algorithm == 1:
            return self.linear_search(target)
        elif algorithm == 2:
            return self.binary_search(target)
        elif algorithm == 3:
            return self.interpolation_search(target)
        elif algorithm == 4:
            return self.exponential_search(target)
        elif algorithm == 5:
            return self.jump_search(target)
        elif algorithm == 6:
            return self.depth_first_search(target)
        elif algorithm == 7:
            return self.breadth_first_search(target)
        elif algorithm == 8:
            return self.astar_search(target)
        else:
            print("Invalid searching algorithm.")
            return -1

    def linear_search(self, target):
        # Linear Search
        start_time = time.time()
        for i in range(len(self.items)):
            if self.items[i] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Linear Search found {target} at index {i}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return i
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Linear Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def binary_search(self, target):
        # Binary Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.items[mid] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Binary Search found {target} at index {mid}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return mid
            elif self.items[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Binary Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def interpolation_search(self, target):
        # Interpolation Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high and self.items[low] <= target <= self.items[high]:
            if low == high:
                if self.items[low] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Interpolation Search found {target} at index {low}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return low
                return -1
            pos = low + ((target - self.items[low]) * (high - low)) // (self.items[high] - self.items[low])
            if self.items[pos] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Interpolation Search found {target} at index {pos}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return pos
            elif self.items[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Interpolation Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def exponential_search(self, target):
        # Exponential Search (Requires sorted array)
        if not self.items:
            return -1
        if self.items[0] == target:
            return 0
        start_time = time.time()
        i = 1
        while i < len(self.items) and self.items[i] <= target:
            i *= 2
        end_time = time.time()
        execution_time = end_time - start_time
        return self.binary_search_recursive(self.items, target, i // 2, min(i, len(self.items) - 1), execution_time)

    def binary_search_recursive(self, arr, target, low, high, total_time):
        if low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                print(f"Exponential Search found {target} at index {mid}.")
                print(f"Execution time: {total_time:.6f} seconds")
                return mid
            elif arr[mid] < target:
                return self.binary_search_recursive(arr, target, mid + 1, high, total_time)
            else:
                return self.binary_search_recursive(arr, target, low, mid - 1, total_time)
        print(f"Exponential Search did not find {target}.")
        print(f"Execution time: {total_time:.6f} seconds")
        return -1

    def jump_search(self, target):
        # Jump Search (Requires sorted array)
        n = len(self.items)
        step = int(n ** 0.5)
        prev = 0
        start_time = time.time()
        while self.items[min(step, n) - 1] < target:
            prev = step
            step += int(n ** 0.5)
            if prev >= n:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Jump Search did not find {target}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return -1
        while self.items[prev] < target:
            prev += 1
        if self.items[prev] == target:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Jump Search found {target} at index {prev}.")
            print(f"Execution time: {execution_time:.6f} seconds")
            return prev
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Jump Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def depth_first_search(self, target):
        # Depth First Search (DFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        stack = []
        stack.append(0)
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"DFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"DFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        stack.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"DFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def breadth_first_search(self, target):
        # Breadth First Search (BFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        queue = []
        queue.append(0)
        while queue:
            node = queue.pop(0)
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"BFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"BFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        queue.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"BFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def astar_search(self, target):
        # A* Search (Astar)
        start_time = time.time()
        open_list = [0]
        closed_list = []
        while open_list:
            node = open_list.pop(0)
            if node not in closed_list:
                closed_list.append(node)
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"A* Search found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if neighbor not in closed_list and neighbor not in open_list:
                        open_list.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"A* Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    # Implement other searching algorithms similarly

    # ... Other existing methods ...

# Rest of your code (Queue, LinkedList, etc.) remains unchanged



    def display(self):
        print("Array:", self.items)

class Stack:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []
        self.type = None

    def push(self, item):
        if len(self.items) < self.max_size:
            self.items.append(item)
        else:
            raise OverflowError("Stack is full")

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def sort(self, algorithm):
        if algorithm == 1:
            self.selection_sort()
        elif algorithm == 2:
            self.merge_sort()
        else:
            print("Invalid sorting algorithm.")

    def sort(self, algorithm):
        if algorithm == 1:
            self.bubble_sort()
        elif algorithm == 2:
            self.selection_sort()
        elif algorithm == 3:
            self.insertion_sort()
        elif algorithm == 4:
            self.merge_sort()
        elif algorithm == 5:
            self.quick_sort()
        elif algorithm == 6:
            self.heap_sort()
        elif algorithm == 7:
            self.radix_sort()
        elif algorithm == 8:
            self.counting_sort()
        else:
            print("Invalid sorting algorithm.")

    def bubble_sort(self):
        # Bubble Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.items[j] > self.items[j + 1]:
                    self.items[j], self.items[j + 1] = self.items[j + 1], self.items[j]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Bubble Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def selection_sort(self):
        # Selection Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.items[j] < self.items[min_idx]:
                    min_idx = j
            self.items[i], self.items[min_idx] = self.items[min_idx], self.items[i]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Selection Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def insertion_sort(self):
        # Insertion Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(1, n):
            key = self.items[i]
            j = i - 1
            while j >= 0 and key < self.items[j]:
                self.items[j + 1] = self.items[j]
                j -= 1
            self.items[j + 1] = key
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Insertion Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def merge_sort(self):
        # Merge Sort
        start_time = time.time()
        if len(self.items) > 1:
            mid = len(self.items) // 2
            left_half = self.items[:mid]
            right_half = self.items[mid:]

            # Recursive calls
            left_half = Array.merge_sort(left_half)
            right_half = Array.merge_sort(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    self.items[k] = left_half[i]
                    i += 1
                else:
                    self.items[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                self.items[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                self.items[k] = right_half[j]
                j += 1
                k += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Merge Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort(self):
        # Quick Sort
        start_time = time.time()
        self.quick_sort_recursive(0, len(self.items) - 1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Quick Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort_recursive(self, low, high):
        if low < high:
            pivot = self.partition(low, high)
            self.quick_sort_recursive(low, pivot - 1)
            self.quick_sort_recursive(pivot + 1, high)

    def partition(self, low, high):
        pivot = self.items[high]
        i = low - 1
        for j in range(low, high):
            if self.items[j] < pivot:
                i += 1
                self.items[i], self.items[j] = self.items[j], self.items[i]
        self.items[i + 1], self.items[high] = self.items[high], self.items[i + 1]
        return i + 1
    def heap_sort(self):
        # Heap Sort
        start_time = time.time()
        n = len(self.items)

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i)

        for i in range(n - 1, 0, -1):
            self.items[i], self.items[0] = self.items[0], self.items[i]
            self.heapify(i, 0)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Heap Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def heapify(self, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.items[left] > self.items[largest]:
            largest = left

        if right < n and self.items[right] > self.items[largest]:
            largest = right

        if largest != i:
            self.items[i], self.items[largest] = self.items[largest], self.items[i]
            self.heapify(n, largest)

    def radix_sort(self):
        # Radix Sort
        start_time = time.time()
        max_num = max(self.items)
        exp = 1

        while max_num // exp > 0:
            self.counting_sort_radix(exp)
            exp *= 10

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Radix Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def counting_sort_radix(self, exp):
        n = len(self.items)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (self.items[i] // exp)
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (self.items[i] // exp)
            output[count[index % 10] - 1] = self.items[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(n):
            self.items[i] = output[i]

    def counting_sort(self):
        # Counting Sort
        start_time = time.time()
        max_num = max(self.items)
        min_num = min(self.items)
        range_of_elements = max_num - min_num + 1

        count = [0] * range_of_elements
        output = [0] * len(self.items)

        for i in range(len(self.items)):
            count[self.items[i] - min_num] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]

        i = len(self.items) - 1
        while i >= 0:
            output[count[self.items[i] - min_num] - 1] = self.items[i]
            count[self.items[i] - min_num] -= 1
            i -= 1

        for i in range(len(self.items)):
            self.items[i] = output[i]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Counting Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")
        
    def search(self, target, algorithm):
        if algorithm == 1:
            return self.linear_search(target)
        elif algorithm == 2:
            return self.binary_search(target)
        elif algorithm == 3:
            return self.interpolation_search(target)
        elif algorithm == 4:
            return self.exponential_search(target)
        elif algorithm == 5:
            return self.jump_search(target)
        elif algorithm == 6:
            return self.depth_first_search(target)
        elif algorithm == 7:
            return self.breadth_first_search(target)
        elif algorithm == 8:
            return self.astar_search(target)
        else:
            print("Invalid searching algorithm.")
            return -1

    def linear_search(self, target):
        # Linear Search
        start_time = time.time()
        for i in range(len(self.items)):
            if self.items[i] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Linear Search found {target} at index {i}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return i
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Linear Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def binary_search(self, target):
        # Binary Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.items[mid] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Binary Search found {target} at index {mid}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return mid
            elif self.items[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Binary Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def interpolation_search(self, target):
        # Interpolation Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high and self.items[low] <= target <= self.items[high]:
            if low == high:
                if self.items[low] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Interpolation Search found {target} at index {low}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return low
                return -1
            pos = low + ((target - self.items[low]) * (high - low)) // (self.items[high] - self.items[low])
            if self.items[pos] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Interpolation Search found {target} at index {pos}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return pos
            elif self.items[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Interpolation Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def exponential_search(self, target):
        # Exponential Search (Requires sorted array)
        if not self.items:
            return -1
        if self.items[0] == target:
            return 0
        start_time = time.time()
        i = 1
        while i < len(self.items) and self.items[i] <= target:
            i *= 2
        end_time = time.time()
        execution_time = end_time - start_time
        return self.binary_search_recursive(self.items, target, i // 2, min(i, len(self.items) - 1), execution_time)

    def binary_search_recursive(self, arr, target, low, high, total_time):
        if low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                print(f"Exponential Search found {target} at index {mid}.")
                print(f"Execution time: {total_time:.6f} seconds")
                return mid
            elif arr[mid] < target:
                return self.binary_search_recursive(arr, target, mid + 1, high, total_time)
            else:
                return self.binary_search_recursive(arr, target, low, mid - 1, total_time)
        print(f"Exponential Search did not find {target}.")
        print(f"Execution time: {total_time:.6f} seconds")
        return -1

    def jump_search(self, target):
        # Jump Search (Requires sorted array)
        n = len(self.items)
        step = int(n ** 0.5)
        prev = 0
        start_time = time.time()
        while self.items[min(step, n) - 1] < target:
            prev = step
            step += int(n ** 0.5)
            if prev >= n:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Jump Search did not find {target}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return -1
        while self.items[prev] < target:
            prev += 1
        if self.items[prev] == target:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Jump Search found {target} at index {prev}.")
            print(f"Execution time: {execution_time:.6f} seconds")
            return prev
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Jump Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def depth_first_search(self, target):
        # Depth First Search (DFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        stack = []
        stack.append(0)
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"DFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"DFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        stack.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"DFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def breadth_first_search(self, target):
        # Breadth First Search (BFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        queue = []
        queue.append(0)
        while queue:
            node = queue.pop(0)
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"BFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"BFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        queue.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"BFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def astar_search(self, target):
        # A* Search (Astar)
        start_time = time.time()
        open_list = [0]
        closed_list = []
        while open_list:
            node = open_list.pop(0)
            if node not in closed_list:
                closed_list.append(node)
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"A* Search found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if neighbor not in closed_list and neighbor not in open_list:
                        open_list.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"A* Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1
        
    def search(self, target, algorithm):
        if algorithm == 1:
            return self.linear_search(target)
        elif algorithm == 2:
            return self.binary_search(target)
        elif algorithm == 3:
            return self.interpolation_search(target)
        elif algorithm == 4:
            return self.exponential_search(target)
        elif algorithm == 5:
            return self.jump_search(target)
        elif algorithm == 6:
            return self.depth_first_search(target)
        elif algorithm == 7:
            return self.breadth_first_search(target)
        elif algorithm == 8:
            return self.astar_search(target)
        else:
            print("Invalid searching algorithm.")
            return -1

    def linear_search(self, target):
        # Linear Search
        start_time = time.time()
        for i in range(len(self.items)):
            if self.items[i] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Linear Search found {target} at index {i}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return i
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Linear Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def binary_search(self, target):
        # Binary Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.items[mid] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Binary Search found {target} at index {mid}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return mid
            elif self.items[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Binary Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def interpolation_search(self, target):
        # Interpolation Search
        start_time = time.time()
        low, high = 0, len(self.items) - 1
        while low <= high and self.items[low] <= target <= self.items[high]:
            if low == high:
                if self.items[low] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Interpolation Search found {target} at index {low}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return low
                return -1
            pos = low + ((target - self.items[low]) * (high - low)) // (self.items[high] - self.items[low])
            if self.items[pos] == target:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Interpolation Search found {target} at index {pos}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return pos
            elif self.items[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Interpolation Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def exponential_search(self, target):
        # Exponential Search (Requires sorted array)
        if not self.items:
            return -1
        if self.items[0] == target:
            return 0
        start_time = time.time()
        i = 1
        while i < len(self.items) and self.items[i] <= target:
            i *= 2
        end_time = time.time()
        execution_time = end_time - start_time
        return self.binary_search_recursive(self.items, target, i // 2, min(i, len(self.items) - 1), execution_time)

    def binary_search_recursive(self, arr, target, low, high, total_time):
        if low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                print(f"Exponential Search found {target} at index {mid}.")
                print(f"Execution time: {total_time:.6f} seconds")
                return mid
            elif arr[mid] < target:
                return self.binary_search_recursive(arr, target, mid + 1, high, total_time)
            else:
                return self.binary_search_recursive(arr, target, low, mid - 1, total_time)
        print(f"Exponential Search did not find {target}.")
        print(f"Execution time: {total_time:.6f} seconds")
        return -1

    def jump_search(self, target):
        # Jump Search (Requires sorted array)
        n = len(self.items)
        step = int(n ** 0.5)
        prev = 0
        start_time = time.time()
        while self.items[min(step, n) - 1] < target:
            prev = step
            step += int(n ** 0.5)
            if prev >= n:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Jump Search did not find {target}.")
                print(f"Execution time: {execution_time:.6f} seconds")
                return -1
        while self.items[prev] < target:
            prev += 1
        if self.items[prev] == target:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Jump Search found {target} at index {prev}.")
            print(f"Execution time: {execution_time:.6f} seconds")
            return prev
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Jump Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def depth_first_search(self, target):
        # Depth First Search (DFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        stack = []
        stack.append(0)
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"DFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"DFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        stack.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"DFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def breadth_first_search(self, target):
        # Breadth First Search (BFS)
        start_time = time.time()
        visited = [False] * len(self.items)
        queue = []
        queue.append(0)
        while queue:
            node = queue.pop(0)
            if not visited[node]:
                visited[node] = True
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"BFS found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if not visited[neighbor] and self.items[neighbor] == target:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"BFS found {target} at index {neighbor}.")
                        print(f"Execution time: {execution_time:.6f} seconds")
                        return neighbor
                    if not visited[neighbor]:
                        queue.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"BFS did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1

    def astar_search(self, target):
        # A* Search (Astar)
        start_time = time.time()
        open_list = [0]
        closed_list = []
        while open_list:
            node = open_list.pop(0)
            if node not in closed_list:
                closed_list.append(node)
                if self.items[node] == target:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"A* Search found {target} at index {node}.")
                    print(f"Execution time: {execution_time:.6f} seconds")
                    return node
                for neighbor in range(len(self.items)):
                    if neighbor not in closed_list and neighbor not in open_list:
                        open_list.append(neighbor)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"A* Search did not find {target}.")
        print(f"Execution time: {execution_time:.6f} seconds")
        return -1


    def display(self):
        print("Stack:", self.items)

class Queue:
    def __init__(self):
        self.items = []
        self.type = None

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return None

    def front(self):
        if not self.is_empty():
            return self.items[0]
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
    def sort(self, algorithm):
        if algorithm == 1:
            self.bubble_sort()
        elif algorithm == 2:
            self.selection_sort()
        elif algorithm == 3:
            self.insertion_sort()
        elif algorithm == 4:
            self.merge_sort()
        elif algorithm == 5:
            self.quick_sort()
        elif algorithm == 6:
            self.heap_sort()
        elif algorithm == 7:
            self.radix_sort()
        elif algorithm == 8:
            self.counting_sort()
        else:
            print("Invalid sorting algorithm.")

    def bubble_sort(self):
        # Bubble Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.items[j] > self.items[j + 1]:
                    self.items[j], self.items[j + 1] = self.items[j + 1], self.items[j]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Bubble Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def selection_sort(self):
        # Selection Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.items[j] < self.items[min_idx]:
                    min_idx = j
            self.items[i], self.items[min_idx] = self.items[min_idx], self.items[i]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Selection Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def insertion_sort(self):
        # Insertion Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(1, n):
            key = self.items[i]
            j = i - 1
            while j >= 0 and key < self.items[j]:
                self.items[j + 1] = self.items[j]
                j -= 1
            self.items[j + 1] = key
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Insertion Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def merge_sort(self):
        # Merge Sort
        start_time = time.time()
        if len(self.items) > 1:
            mid = len(self.items) // 2
            left_half = self.items[:mid]
            right_half = self.items[mid:]

            # Recursive calls
            left_half = Array.merge_sort(left_half)
            right_half = Array.merge_sort(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    self.items[k] = left_half[i]
                    i += 1
                else:
                    self.items[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                self.items[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                self.items[k] = right_half[j]
                j += 1
                k += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Merge Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort(self):
        # Quick Sort
        start_time = time.time()
        self.quick_sort_recursive(0, len(self.items) - 1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Quick Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort_recursive(self, low, high):
        if low < high:
            pivot = self.partition(low, high)
            self.quick_sort_recursive(low, pivot - 1)
            self.quick_sort_recursive(pivot + 1, high)

    def partition(self, low, high):
        pivot = self.items[high]
        i = low - 1
        for j in range(low, high):
            if self.items[j] < pivot:
                i += 1
                self.items[i], self.items[j] = self.items[j], self.items[i]
        self.items[i + 1], self.items[high] = self.items[high], self.items[i + 1]
        return i + 1
    def heap_sort(self):
        # Heap Sort
        start_time = time.time()
        n = len(self.items)

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i)

        for i in range(n - 1, 0, -1):
            self.items[i], self.items[0] = self.items[0], self.items[i]
            self.heapify(i, 0)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Heap Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def heapify(self, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.items[left] > self.items[largest]:
            largest = left

        if right < n and self.items[right] > self.items[largest]:
            largest = right

        if largest != i:
            self.items[i], self.items[largest] = self.items[largest], self.items[i]
            self.heapify(n, largest)

    def radix_sort(self):
        # Radix Sort
        start_time = time.time()
        max_num = max(self.items)
        exp = 1

        while max_num // exp > 0:
            self.counting_sort_radix(exp)
            exp *= 10

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Radix Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def counting_sort_radix(self, exp):
        n = len(self.items)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (self.items[i] // exp)
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (self.items[i] // exp)
            output[count[index % 10] - 1] = self.items[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(n):
            self.items[i] = output[i]

    def counting_sort(self):
        # Counting Sort
        start_time = time.time()
        max_num = max(self.items)
        min_num = min(self.items)
        range_of_elements = max_num - min_num + 1

        count = [0] * range_of_elements
        output = [0] * len(self.items)

        for i in range(len(self.items)):
            count[self.items[i] - min_num] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]

        i = len(self.items) - 1
        while i >= 0:
            output[count[self.items[i] - min_num] - 1] = self.items[i]
            count[self.items[i] - min_num] -= 1
            i -= 1

        for i in range(len(self.items)):
            self.items[i] = output[i]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Counting Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")
    


    def search(self, item, algorithm):
        if algorithm == 1:
            return self.linear_search(item)
        elif algorithm == 2:
            return self.binary_search(item)
        else:
            print("Invalid searching algorithm.")
            return False

    def linear_search(self, item):
        for i, value in enumerate(self.items):
            if value == item:
                return i
        return -1

    def binary_search(self, item):
        left, right = 0, len(self.items) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.items[mid] == item:
                return mid
            elif self.items[mid] < item:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def display(self):
        print("Queue:", self.items)

class Deque:
    def __init__(self):
        self.items = []
        self.type = None

    def is_empty(self):
        return len(self.items) == 0

    def add_front(self, item):
        self.items.insert(0, item)

    def add_rear(self, item):
        self.items.append(item)

    def remove_front(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return None

    def remove_rear(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return None

    def size(self):
        return len(self.items)

    def peek_front(self):
        if not self.is_empty():
            return self.items[0]
        else:
            return None

    def peek_rear(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            return None
    
    def sort(self, algorithm):
        if algorithm == 1:
            self.bubble_sort()
        elif algorithm == 2:
            self.selection_sort()
        elif algorithm == 3:
            self.insertion_sort()
        elif algorithm == 4:
            self.merge_sort()
        elif algorithm == 5:
            self.quick_sort()
        elif algorithm == 6:
            self.heap_sort()
        elif algorithm == 7:
            self.radix_sort()
        elif algorithm == 8:
            self.counting_sort()
        else:
            print("Invalid sorting algorithm.")

    def bubble_sort(self):
        # Bubble Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.items[j] > self.items[j + 1]:
                    self.items[j], self.items[j + 1] = self.items[j + 1], self.items[j]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Bubble Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def selection_sort(self):
        # Selection Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.items[j] < self.items[min_idx]:
                    min_idx = j
            self.items[i], self.items[min_idx] = self.items[min_idx], self.items[i]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Selection Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def insertion_sort(self):
        # Insertion Sort
        start_time = time.time()
        n = len(self.items)
        for i in range(1, n):
            key = self.items[i]
            j = i - 1
            while j >= 0 and key < self.items[j]:
                self.items[j + 1] = self.items[j]
                j -= 1
            self.items[j + 1] = key
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Insertion Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def merge_sort(self):
        # Merge Sort
        start_time = time.time()
        if len(self.items) > 1:
            mid = len(self.items) // 2
            left_half = self.items[:mid]
            right_half = self.items[mid:]

            # Recursive calls
            left_half = Array.merge_sort(left_half)
            right_half = Array.merge_sort(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    self.items[k] = left_half[i]
                    i += 1
                else:
                    self.items[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                self.items[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                self.items[k] = right_half[j]
                j += 1
                k += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Merge Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort(self):
        # Quick Sort
        start_time = time.time()
        self.quick_sort_recursive(0, len(self.items) - 1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Quick Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def quick_sort_recursive(self, low, high):
        if low < high:
            pivot = self.partition(low, high)
            self.quick_sort_recursive(low, pivot - 1)
            self.quick_sort_recursive(pivot + 1, high)

    def partition(self, low, high):
        pivot = self.items[high]
        i = low - 1
        for j in range(low, high):
            if self.items[j] < pivot:
                i += 1
                self.items[i], self.items[j] = self.items[j], self.items[i]
        self.items[i + 1], self.items[high] = self.items[high], self.items[i + 1]
        return i + 1
    def heap_sort(self):
        # Heap Sort
        start_time = time.time()
        n = len(self.items)

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i)

        for i in range(n - 1, 0, -1):
            self.items[i], self.items[0] = self.items[0], self.items[i]
            self.heapify(i, 0)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Heap Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def heapify(self, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.items[left] > self.items[largest]:
            largest = left

        if right < n and self.items[right] > self.items[largest]:
            largest = right

        if largest != i:
            self.items[i], self.items[largest] = self.items[largest], self.items[i]
            self.heapify(n, largest)

    def radix_sort(self):
        # Radix Sort
        start_time = time.time()
        max_num = max(self.items)
        exp = 1

        while max_num // exp > 0:
            self.counting_sort_radix(exp)
            exp *= 10

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Radix Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def counting_sort_radix(self, exp):
        n = len(self.items)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (self.items[i] // exp)
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (self.items[i] // exp)
            output[count[index % 10] - 1] = self.items[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(n):
            self.items[i] = output[i]

    def counting_sort(self):
        # Counting Sort
        start_time = time.time()
        max_num = max(self.items)
        min_num = min(self.items)
        range_of_elements = max_num - min_num + 1

        count = [0] * range_of_elements
        output = [0] * len(self.items)

        for i in range(len(self.items)):
            count[self.items[i] - min_num] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]

        i = len(self.items) - 1
        while i >= 0:
            output[count[self.items[i] - min_num] - 1] = self.items[i]
            count[self.items[i] - min_num] -= 1
            i -= 1

        for i in range(len(self.items)):
            self.items[i] = output[i]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Counting Sort completed.")
        print(f"Execution time: {execution_time:.6f} seconds")

    def search(self, item, algorithm):
        if algorithm == 1:
            return self.linear_search(item)
        elif algorithm == 2:
            return self.binary_search(item)
        else:
            print("Invalid searching algorithm.")
            return False

    def linear_search(self, item):
        for i, value in enumerate(self.items):
            if value == item:
                return i
        return -1

    def binary_search(self, item):
        left, right = 0, len(self.items) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.items[mid] == item:
                return mid
            elif self.items[mid] < item:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def display(self):
        print("Deque:", self.items)

class PriorityQueue:
    def __init__(self):
        self.items = []
        self.type = None

    def is_empty(self):
        return len(self.items) == 0

    def put(self, item, priority):
        self.items.append((item, priority))

    def get(self):
        if not self.is_empty():
            highest_priority_item = min(self.items, key=lambda x: x[1])
            self.items.remove(highest_priority_item)
            return highest_priority_item[0]
        else:
            return None

    def display(self):
        print("Queue:", self.items)


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.type = None

    def insert(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        if self.head is None:
            return
        if self.head.data == data:
            self.head = self.head.next
            return

        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def search(self, data):
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class AVLNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1

class RedBlackNode:
    def __init__(self, data, color="RED"):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None
        self.color = color  # "RED" or "BLACK"

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        self.root = self._insert_recursive(self.root, data)

    def _insert_recursive(self, current_node, data):
        if current_node is None:
            return TreeNode(data)
        if data < current_node.data:
            current_node.left = self._insert_recursive(current_node.left, data)
        else:
            current_node.right = self._insert_recursive(current_node.right, data)
        return current_node

    def search(self, data):
        return self._search_recursive(self.root, data)

    def _search_recursive(self, current_node, data):
        if current_node is None or current_node.data == data:
            return current_node is not None
        if data < current_node.data:
            return self._search_recursive(current_node.left, data)
        return self._search_recursive(current_node.right, data)

    def inorder_traversal(self):
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, current_node, result):
        if current_node:
            self._inorder_recursive(current_node.left, result)
            result.append(current_node.data)
            self._inorder_recursive(current_node.right, result)

# AVL Tree implementation (Balanced Binary Search Tree)
class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        self.root = self._insert_recursive(self.root, data)

    def _insert_recursive(self, current_node, data):
        if current_node is None:
            return AVLNode(data)
        if data < current_node.data:
            current_node.left = self._insert_recursive(current_node.left, data)
        else:
            current_node.right = self._insert_recursive(current_node.right, data)

        # Update height
        current_node.height = 1 + max(self._get_height(current_node.left), self._get_height(current_node.right))

        # Check balance and rotate if needed
        return self._balance(current_node, data)

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def _balance(self, node, data):
        # Calculate balance factor
        balance = self._get_balance(node)

        # Left Heavy (Right Rotation)
        if balance > 1 and data < node.left.data:
            return self._rotate_right(node)

        # Right Heavy (Left Rotation)
        if balance < -1 and data > node.right.data:
            return self._rotate_left(node)

        # Left Right Heavy (Left-Right Rotation)
        if balance > 1 and data > node.left.data:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right Left Heavy (Right-Left Rotation)
        if balance < -1 and data < node.right.data:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _get_balance(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

# Red-Black Tree implementation
class RedBlackTree: #deltha me da
    def __init__(self):
        self.nil = RedBlackNode(data=None, color="BLACK")
        self.root = self.nil

    def insert(self, data):
        new_node = RedBlackNode(data, color="RED")
        self._insert_recursive(new_node)

    def _insert_recursive(self, new_node):
        current_node = self.root
        parent_node = None

        while current_node != self.nil:
            parent_node = current_node
            if new_node.data < current_node.data:
                current_node = current_node.left
            else:
                current_node = current_node.right

        new_node.parent = parent_node
        if parent_node is None:
            self.root = new_node
        elif new_node.data < parent_node.data:
            parent_node.left = new_node
        else:
            parent_node.right = new_node

        if new_node.parent is None:
            new_node.color = "BLACK"
            return

        if new_node.parent.parent is None:
            return

        self._insert_fixup(new_node)

    def _insert_fixup(self, new_node):
        while new_node.parent.color == "RED":
            if new_node.parent == new_node.parent.parent.left:
                y = new_node.parent.parent.right
                if y.color == "RED":
                    new_node.parent.color = "BLACK"
                    y.color = "BLACK"
                    new_node.parent.parent.color = "RED"
                    new_node = new_node.parent.parent
                else:
                    if new_node == new_node.parent.right:
                        new_node = new_node.parent
                        self._left_rotate(new_node)
                    new_node.parent.color = "BLACK"
                    new_node.parent.parent.color = "RED"
                    self._right_rotate(new_node.parent.parent)
            else:
                y = new_node.parent.parent.left
                if y.color == "RED":
                    new_node.parent.color = "BLACK"
                    y.color = "BLACK"
                    new_node.parent.parent.color = "RED"
                    new_node = new_node.parent.parent
                else:
                    if new_node == new_node.parent.left:
                        new_node = new_node.parent
                        self._right_rotate(new_node)
                    new_node.parent.color = "BLACK"
                    new_node.parent.parent.color = "RED"
                    self._left_rotate(new_node.parent.parent)

        self.root.color = "BLACK"
        

    def _left_rotate(self, node):
        y = node.right
        node.right = y.left
        if y.left != self.nil:
            y.left.parent = node
        y.parent = node.parent
        if node.parent is None:
            self.root = y
        elif node == node.parent.left:
            node.parent.left = y
        else:
            node.parent.right = y
        y.left = node
        node.parent = y

    def _right_rotate(self, node):
        x = node.left
        node.left = x.right
        if x.right != self.nil:
            x.right.parent = node
        x.parent = node.parent
        if node.parent is None:
            self.root = x
        elif node == node.parent.right:
            node.parent.right = x
        else:
            node.parent.left = x
        x.right = node
        node.parent = x
    
    def inorder_traversal(self, node):
        if node != self.nil:
            self.inorder_traversal(node.left)
            print(f"{node.data}({node.color})", end=" ")
            self.inorder_traversal(node.right)

    def search(self, data):
        return self._search_recursive(self.root, data)

    def _search_recursive(self, node, data):
        if node == self.nil or node.data == data:
            return node
        if data < node.data:
            return self._search_recursive(node.left, data)
        return self._search_recursive(node.right, data)


class Graph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def add_edge(self, vertex1, vertex2, weight=None):
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1][vertex2] = weight
            if not self.directed:
                self.graph[vertex2][vertex1] = weight

    def display(self):
        for vertex, edges in self.graph.items():
            print(f"{vertex}: {edges}")


# Define a class called HashTable to represent a hash table data structure.
class HashTable:
    # Initialize the hash table with a specified maximum size.
    def __init__(self, max_size):
        self.max_size = max_size  # Maximum number of slots in the hash table
        self.size = 0            # Current number of elements stored in the hash table
        self.table = [None] * max_size  # Initialize an empty list (array) of slots

    # Define a method for hashing a given key to determine the slot index.
    def hash(self, key):
        return hash(key) % self.max_size  # Calculate the index by taking the remainder of the key's hash value.

    # Define a method to insert a key-value pair into the hash table.
    def put(self, key, value):
        # Check if the hash table is already full.
        if self.size >= self.max_size:
            print("The hash table is full. Cannot insert more items.")
            return  # Exit the method if the table is full.

        index = self.hash(key)  # Compute the index where the key-value pair should be stored.
        
        # If the slot at the computed index is empty, create a list to hold key-value pairs.
        if self.table[index] is None:
            self.table[index] = []

        # Check if the key already exists in the list, and update the value if it does.
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)  # Update the existing key-value pair.
                return
        
        # If the key is not found in the list, append a new key-value pair to the list.
        self.table[index].append((key, value))
        self.size += 1  # Increment the size of the hash table to reflect the addition.

    # Define a method to retrieve the value associated with a given key.
    def get(self, key):
        index = self.hash(key)  # Compute the index where the key should be located.
        if self.table[index] is not None:  # Check if there is a list at the computed index.
            for k, v in self.table[index]:  # Iterate through key-value pairs in the list.
                if k == key:
                    return v  # Return the value if the key is found.
        return None  # Return None if the key is not found in the hash table.

def main():
    while True:
        print("Available data structures:")
        print("1. Array")
        print("2. Stack")
        print("3. Queue")
        print("4. LinkedList")
        print("5. Tree")
        print("6. Graph")
        print("7. Hash table")
        print("8. Exit")
        try:
            choice = int(input("Enter the number of your choice: "))
        except:
            print("Invalid input!\n")
            continue
        if choice == 1:
            while True:
                print("What do you want?")
                print("1. Defenation")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter the choice: "))
                except:
                    print("Invalid input!\n")
                    continue
                if select == 1:
                    print("\nArray:\n\tAn array is a collection of elements, each identified by an index or a key.")
                elif select == 2:
                    print(""" \nApplication:\n\tsimple application of array that it is used in searching algorithms like linear_search
    And the application code is:
    def linear_search(arr, target):
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
                          
    """)
                elif select == 3:
                    while True:
                        try:
                            max_elements = int(input("Enter the size of the array: "))
                            break
                        except:
                            print("Invalid input!")
                    array = Array(max_elements)
                    while True:
                        print("types of items: ")
                        print("1. Integer")
                        print("2. Float")
                        print("3. String")
                        try:
                            selected_type = int(input("Select the type of items: "))
                        except:
                            print("Invalid input!\n")
                            continue
                        if selected_type == 1:
                            array.type = "INT"
                            break
                        elif selected_type == 2:
                            array.type = "Float"
                            break
                        elif selected_type == 3:
                            array.type = "String"
                            break
                        
                        else:
                            print("Invalid input")
                    while True:
                        try:
                            num_elements = int(input("Enter the number of elements you want to add: "))
                            break
                        except:
                            print("Invalid input!\n")
                            continue
                    for i in range(num_elements):
                        if array.type == "INT":
                            while True:
                                try:
                                    item = int(input(f"Enter element {i + 1}: "))
                                    array.insert(item)
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.")

                        elif array.type == "Float":
                            while True:
                                try:
                                    item = float(input(f"Enter element {i + 1}: "))
                                    array.insert(item)
                                    break  # Exit the loop if a valid float is provided
                                except ValueError:
                                    print("Invalid input. Please enter a float.")

                        elif array.type == "String":
                            while True:
                                try:
                                    item = input(f"Enter element {i + 1}: ")
                                    array.insert(item)
                                    break
                                except ValueError:
                                    print("Invalid input. Please enter a string.")

                        else :
                            print("The type of inputs is not selected.")

                    while True:
                        print("\nArray Operations:")
                        print("1. Insert Element")
                        print("2. Delete Element")
                        print("3. Sort Array")
                        print("4. Search Element")
                        print("5. Display Array")
                        print("6. Back")
                        print("7. Exit")
                        while True:
                            try:
                                operation = int(input("Enter your choice: "))
                                break
                            except:
                                print("Invalid input. Please enter a number.")

                        if operation == 1:
                            if (len(array.items))<array.max_size:
                                if array.type == "INT":
                                    try:
                                        item = int(input("Enter item to insert: "))
                                        array.insert(item)
                                        print("Updated array:", array.items)
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue

                                elif array.type == "Float":
                                    try:
                                        item = float(input("Enter item to insert: "))
                                        array.insert(item)
                                        print("Updated array:", array.items)
                                    except:
                                        print("Invalid input. Please enter a float.")
                                        continue

                                elif array.type == "String":
                                    item = input("Enter item to insert: ")
                                    array.insert(item)
                                    print("Updated array:", array.items)
                                
                            else:
                                print("Sorry array is full!!!")

                        elif operation == 2:
                            if array.type == "INT":
                                try:
                                    item = int(input("Enter item to delete: "))
                                    array.delete(item)
                                except:
                                    print("Invalid input. Please enter an integer.")
                                    continue
                            elif array.type == "Float":
                                try:
                                    item = float(input("Enter item to delete: "))
                                    array.delete(item)
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                            elif array.type == "String":
                                item = input("Enter item to delete: ")
                                array.delete(item)
                            print("Updated array:", array.items)

                        elif operation == 3:
                            print("Sorting algorithms:")
                            print("1. Bubble Sort")
                            print("2. selection_sort")
                            print("3.insertion_sort")
                            print("4. merge_sort")
                            print("5. quick_sort")
                            print("6. heap_sort")
                            print("7. radix_sort")
                            print("8. counting_sort")
                            try:
                                sort_algorithm = int(input("Enter the number of the sorting algorithm: "))
                                array.sort(sort_algorithm)
                            except:
                                print("Invalid input. Please enter an integer.")
                                continue
                        elif operation == 4:
                            if array.type == "INT":
                                try:
                                    item = int(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter an integer.")
                                    continue
                            elif array.type == "Float":
                                try:
                                    item = float(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue

                            elif array.type == "String":
                                item = input("Enter item to search: ")

                            print("Searching algorithms:")
                            print("1. Linear Search")
                            print("2. Binary Search")
                            print("3. interpolation_search")
                            print("4. exponential_search")
                            print("5. jump_search")
                            print("6. depth_first_search")
                            print("7. breadth_first_search")
                            print("8. astar_search")
                            while True:
                                try:
                                    search_algorithm = int(input("Enter the number of the searching algorithm: "))
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.")
                            result = array.search(item, search_algorithm)
                            if result != -1:
                                print(f"Item {item} found at index {result}.")
                            else:
                                print(f"Item {item} not found in the array.")
                        elif operation == 5:
                            array.display()
                        elif operation == 6:
                            break
                        elif operation == 7:
                            exit()
                        else:
                            print("Invalid choice. Please try again.")
                elif select == 4:
                    break
                elif select == 5:
                    exit()
                else:
                    print("Invalid input")

        elif choice == 2:
            while True:
                print("What do you want?")
                print("1. Defenation")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter the choice: "))
                except:
                    print("Invalid input. Please enter an integer.")
                    continue
                if select == 1:
                    print("\nStack:\n\tA stack is a linear data structure that follows the Last In First Out (LIFO) principle.")
                elif select == 2:
                    print("""simple application of stack is to check if parentheses in an expression are balanced.
            and the code of balancing parantheses is :
            def is_balanced(expression):
        stack = []
        brackets = {"(": ")", "[": "]", "{": "}"}

        for char in expression:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack or char != brackets[stack.pop()]:
                    return False

        return not stack

    balanced = is_balanced("{[()]}")""")
                elif select == 3:
                    try:
                        max_elements = int(input("Enter the maximum number of elements: "))
                    except:
                        print("Invalid input. Please enter an integer.")
                        continue
                    stack = Stack(max_elements)
                    while True:
                        print("types of items: ")
                        print("1. Integer")
                        print("2. Float")
                        print("3. String")
                        try:
                            selected_type = int(input("Select the type of items: "))
                        except:
                            print("Invalid input. Please enter an integer.\n")
                            continue
                        if selected_type == 1:
                            stack.type = "INT"
                            break
                        elif selected_type == 2:
                            stack.type = "Float"
                            break
                        elif selected_type == 3:
                            stack.type = "String"
                            break
                        else:
                            print("Invalid input")
                    while True:
                        try:
                            num_elements = int(input(f"Enter the number of elements up to {max_elements}, you want to add now: "))
                            break
                        except:
                            print("Invalid input. Please enter an integer.")
                    for i in range(num_elements):
                        if stack.type == "INT":
                            while True:
                                try:
                                    item = int(input(f"Enter element {i + 1}: "))
                                    stack.push(item)
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.")
                            
                        elif stack.type == "Float":
                            while True:
                                try:
                                    item = float(input(f"Enter element {i + 1}: "))
                                    stack.push(item)
                                    break
                                except:
                                    print("Invalid input. Please enter a float.")
                        elif stack.type == "String":
                            item = input(f"Enter element {i + 1}: ")
                            stack.push(item)
                        else :
                            print("The type of inputs is not selected.")

                    while True:
                        print("\nStack Operations:")
                        print("1. Push Element")
                        print("2. Pop Element")
                        print("3. Peek Element")
                        print("4. Sort Stack")
                        print("5. Search Element")
                        print("6. Display Stack")
                        print("7. Back")
                        print("8. Exit")
                        
                        try:
                            operation = int(input("Enter your choice: "))
                        except:
                            print("Invalid input. Please enter an integer.")
                            continue
                        if operation == 1:
                            if stack.type == "INT":
                                try:
                                    item = int(input("Enter item to push: "))
                                    stack.push(item)
                                    print("Updated stack:", stack.items)
                                except:
                                    print("Invalid input. Please enter an integer.")
                            elif stack.type == "Float":
                                try:
                                    item = float(input("Enter item to push: "))
                                    stack.push(item)
                                    print("Updated stack:", stack.items)
                                except:
                                    print("Invalid input. Please enter a float.")
                            elif stack.type == "String":
                                item = input("Enter item to push: ")
                                stack.push(item)
                                print("Updated stack:", stack.items)
                
                        elif operation == 2:
                            popped_item = stack.pop()
                            if popped_item is not None:
                                print(f"Popped item: {popped_item}")
                                print("Updated stack:", stack.items)
                            else:
                                print("Stack is empty. Cannot pop.")
                        elif operation == 3:
                            peeked_item = stack.peek()
                            if peeked_item is not None:
                                print(f"Peeked item: {peeked_item}")
                            else:
                                print("Stack is empty. Cannot peek.")
                        elif operation == 4:
                            print("Sorting algorithms:")
                            print("1. bubble_sort")
                            print("2. selection_sort")
                            print("3. insertion_sort")
                            print("4. Merge Sort")
                            print("5. quick_sort")
                            #print("6. heap_sort")
                            #print("7. radix_sort")
                            #print("8. counting_sort")
                            try:
                                sort_algorithm = int(input("Enter the number of the sorting algorithm: "))
                                stack.sort(sort_algorithm)
                            except:
                                print("Invalid input. Please enter an integer.")
                        elif operation == 5:
                            if stack.type == "INT":
                                try:
                                    item = int(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter an integer.")

                            elif stack.type == "Float":
                                try:
                                    item = float(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                            elif stack.type == "String":
                                item = input("Enter item to search: ")

                            print("Searching algorithms:")
                            print("1. Linear Search")
                            print("2. Binary Search")
                            print("3. interpolation_search")
                            print("4. exponential_search")
                            print("5. jump_search")
                            print("6. depth_first_search")
                            print("7. breadth_first_search")
                            print("8. astar_search")
                            
                            try:
                                search_algorithm = int(input("Enter the number of the searching algorithm: "))
                                result = stack.search(item, search_algorithm)
                            except:
                                print("Invalid input. Please enter an integer.")
                            if result != -1:
                                print(f"Item {item} found at index {result}.")
                            else:
                                print(f"Item {item} not found in the stack.")
                        elif operation == 6:
                            stack.display()
                        elif operation == 7:
                            break
                        elif operation == 8:
                            exit()
                        else:
                            print("Invalid choice. Please try again.")
                elif select == 4:
                    break
                elif select == 5:
                    exit()
                else:
                    print("Invalid input")

        elif choice == 3:
            while True:
                print("What do you want?")
                print("1. Defenation")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter the choice: "))
                except:
                    print("Invalid input. Please enter an integer.\n")
                    continue
                if select == 1:
                    print("\nQueue Definition:\n\tA queue is a linear data structure that follows the First In First Out (FIFO) principle.")
                elif select == 2:
                    print("""\n A simple application of queue is that it is used in call center system
            and the code is :
            from queue import Queue

    call_queue = Queue()
    call_queue.put("Call 1")
    call_queue.put("Call 2")
    next_call = call_queue.get()
    """)
                elif select == 3:
                    while True:
                        print("\nTypes of queue: ")
                        print("1. Queue")
                        print("2. Deque")
                        print("3. PriorityQueue")
                        print("4. Back")
                        print("5. Exit")
                        try:
                            select = int(input("Enter the choice: "))
                        except:
                            print("Invalid input. Please enter an integer.\n")
                            continue
                        if select == 1:
                            queue = Queue()
                            print("types of items: ")
                            print("1. Integer")
                            print("2. Float")
                            print("3. String")
                            try:
                                selected_type = int(input("Select the type of items: "))
                            except:
                                print("Invalid input. Please enter an integer.")
                                continue
                            if selected_type == 1:
                                queue.type = "INT"

                            elif selected_type == 2:
                                queue.type = "Float"
                                
                            elif selected_type == 3:
                                queue.type = "String"
                                
                            else:
                                print("Invalid input")

                            while True:     
                                try:
                                    num_elements = int(input("Enter the number of elements you want to add: "))
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.\n")
                            for i in range(num_elements):
                                if queue.type == "INT":
                                    while True:
                                        try:
                                            item = int(input(f"Enter element {i + 1}: "))
                                            queue.enqueue(item)
                                            break
                                        except:
                                            print("Invalid input. Please enter an integer.")

                                elif queue.type == "Float":
                                    while True:
                                        try:
                                            item = float(input(f"Enter element {i + 1}: "))
                                            queue.enqueue(item)
                                            break
                                        except:
                                            print("Invalid input. Please enter a float.")

                                elif queue.type == "String":
                                    item = input(f"Enter element {i + 1}: ")
                                    queue.enqueue(item)
                            while True:
                                print("\nQueue Operations:")
                                print("1. Enqueue Element")
                                print("2. Dequeue Element")
                                print("3. Front Element")
                                print("4. Sort Queue")
                                print("5. Search Element")
                                print("6. Display Queue")
                                print("7. Back")
                                print("8. Exit")
                                try:
                                    operation = int(input("Enter your choice: "))
                                    break
                                except:
                                    print("please enter a number.")
                                    continue
                            if operation == 1:
                                if queue.type == "INT":
                                    try:
                                        item = int(input("Enter item to enqueue: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    queue.enqueue(item)
                                elif queue.type == "Float":
                                    try:
                                        item = float(input("Enter item to enqueue: "))
                                    except:
                                        print("Invalid input. Please enter a float.")
                                        continue
                                    queue.enqueue(item)
                                elif queue.type == "String":
                                    item = input("Enter item to enqueue: ")
                                    queue.enqueue(item)
                                print("Updated queue:", queue.items)
                            elif operation == 2:
                                dequeued_item = queue.dequeue()
                                if dequeued_item is not None:
                                    print(f"Dequeued item: {dequeued_item}")
                                    print("Updated queue:", queue.items)
                                else:
                                    print("Queue is empty. Cannot dequeue.")
                            elif operation == 3:
                                front_item = queue.front()
                                if front_item is not None:
                                    print(f"Front item: {front_item}")
                                else:
                                    print("Queue is empty. No front item.")
                            elif operation == 4:
                                print("Sorting algorithms:")
                                print("1. bubble_sort")
                                print("2. selection_sort")
                                print("3. insertion_sort")
                                print("4. Merge Sort")
                                print("5. quick_sort")
                                #print("6. heap_sort")
                                #print("7. radix_sort")
                                #print("8. counting_sort")
                                try:
                                    sort_algorithm = int(input("Enter the number of the sorting algorithm: "))
                                except:
                                    print("Invalid input. Please enter an integer.")
                                queue.sort(sort_algorithm)
                            elif operation == 5:
                                if queue.type == "INT":
                                    item = int(input("Enter item to search: "))
                                elif queue.type == "Float":
                                    item = float(input("Enter item to search: "))
                                elif queue.type == "String":
                                    item = input("Enter item to search: ")
                                
                                    
                                print("Searching algorithms:")
                                print("1. Linear Search")
                                print("2. Binary Search")
                                print("3. interpolation_search")
                                print("4. exponential_search")
                                print("5. jump_search")
                                print("6. depth_first_search")
                                print("7. breadth_first_search")
                                print("8. astar_search")
                                search_algorithm = int(input("Enter the number of the searching algorithm: "))
                                result = queue.search(item, search_algorithm)
                                if result != -1:
                                    print(f"Item {item} found at index {result}.")
                                else:
                                    print(f"Item {item} not found in the queue.")
                            elif operation == 6:
                                queue.display()
                            elif operation == 7:
                                break
                            elif operation == 8:
                                exit()
                            else:
                                print("Invalid choice. Please try again.")
                        
                        elif select == 2:
                            queue = Deque()
                            print("types of items: ")
                            print("1. Integer")
                            print("2. Float")
                            print("3. String")
                            try:
                                selected_type = int(input("Select the type of items: "))
                            except:
                                print("Invalid input. Please enter an integer.")
                                continue
                            if selected_type == 1:
                                queue.type = "INT"
                                
                            elif selected_type == 2:
                                queue.type = "Float"
                                
                            elif selected_type == 3:
                                queue.type = "String"
                                
                            else:
                                print("Invalid input")

                            while True:     
                                try:
                                    num_elements = int(input("Enter the number of elements you want to add: "))
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.\n")
                            for i in range(num_elements):
                                if queue.type == "INT":
                                    while True:
                                        try:
                                            item = int(input(f"Enter element {i + 1}: "))
                                            queue.add_rear(item)
                                            break
                                        except:
                                            print("Invalid input. Please enter an integer.")

                                elif queue.type == "Float":
                                    while True:
                                        try:
                                            item = float(input(f"Enter element {i + 1}: "))
                                            queue.add_rear(item)
                                            break
                                        except:
                                            print("Invalid input. Please enter a float.")

                                elif queue.type == "String":
                                    item = input(f"Enter element {i + 1}: ")
                                    queue.add_rear(item)
                            while True:
                                print("\nQueue Operations:")
                                print("1. Add-rear Element")
                                print("2. Add-front Element")
                                print("3. Remove-front Element")
                                print("4. Remove-rear Element")
                                print("5. Front Element")
                                print("6. Rear Element")
                                print("7. Sort Queue")
                                print("8. Search Element")
                                print("9. Display Queue")
                                print("10. Back")
                                print("11. Exit")
                                try:
                                    operation = int(input("Enter your choice: "))
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.\n")
                                    continue
                            if operation == 1:
                                if queue.type == "INT":
                                    try:
                                        item = int(input("Enter item to add_rear: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    queue.add_rear(item)
                                elif queue.type == "Float":
                                    try:
                                        item = float(input("Enter item to add_rear: "))
                                    except:
                                        print("Invalid input. Please enter a float.")
                                        continue
                                    queue.add_rear(item)
                                elif queue.type == "String":
                                    item = input("Enter item to add_rear: ")
                                    queue.add_rear(item)
                                print("Updated queue:", queue.items)
                            elif operation == 2:
                                if queue.type == "INT":
                                    try:
                                        item = int(input("Enter item to add_front: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    queue.add_front(item)
                                elif queue.type == "Float":
                                    try:
                                        item = float(input("Enter item to add_front: "))
                                    except:
                                        print("Invalid input. Please enter a float.")
                                        continue
                                    queue.add_front(item)
                                elif queue.type == "String":
                                    item = input("Enter item to add_front: ")
                                    queue.add_front(item)
                                print("Updated queue:", queue.items)
                            elif operation == 3:
                                dequeued_item = queue.remove_front()
                                if dequeued_item is not None:
                                    print(f"Dequeued item: {dequeued_item}")
                                    print("Updated queue:", queue.items)
                                else:
                                    print("Queue is empty. Cannot dequeue.")
                            elif operation == 4:
                                dequeued_item = queue.remove_rear()
                                if dequeued_item is not None:
                                    print(f"Dequeued item: {dequeued_item}")
                                    print("Updated queue:", queue.items)
                                else:
                                    print("Queue is empty. Cannot dequeue.")
                            elif operation == 5:
                                front_item = queue.peek_front()
                                if front_item is not None:
                                    print(f"Front item: {front_item}")
                                else:
                                    print("Queue is empty. No front item.")
                            elif operation == 6:
                                Rear_item = queue.peek_rear()
                                if Rear_item is not None:
                                    print(f"Rear item: {Rear_item}")
                                else:
                                    print("Queue is empty. No front item.")
                            elif operation == 7:
                                print("Sorting algorithms:")
                                print("1. bubble_sort")
                                print("2. selection_sort")
                                print("3. insertion_sort")
                                print("4. Merge Sort")
                                print("5. quick_sort")
                                #print("6. heap_sort")
                                #print("7. radix_sort")
                                #print("8. counting_sort")
                                try:
                                    sort_algorithm = int(input("Enter the number of the sorting algorithm: "))
                                except:
                                    print("Invalid input. Please enter an integer.")
                                queue.sort(sort_algorithm)
                            elif operation == 8:
                                if queue.type == "INT":
                                    item = int(input("Enter item to search: "))
                                elif queue.type == "Float":
                                    item = float(input("Enter item to search: "))
                                elif queue.type == "String":
                                    item = input("Enter item to search: ")
                                
                                    
                                print("Searching algorithms:")
                                print("1. Linear Search")
                                print("2. Binary Search")
                                print("3. interpolation_search")
                                print("4. exponential_search")
                                print("5. jump_search")
                                print("6. depth_first_search")
                                print("7. breadth_first_search")
                                print("8. astar_search")
                                search_algorithm = int(input("Enter the number of the searching algorithm: "))
                                result = queue.search(item, search_algorithm)
                                if result != -1:
                                    print(f"Item {item} found at index {result}.")
                                else:
                                    print(f"Item {item} not found in the queue.")
                            elif operation == 9:
                                queue.display()
                            elif operation == 10:
                                break
                            elif operation == 11:
                                exit()
                            else:
                                print("Invalid input")
                        elif select == 3:
                            queue = PriorityQueue()
                            print("types of items: ")
                            print("1. Integer")
                            print("2. Float")
                            print("3. String")
                            try:
                                selected_type = int(input("Select the type of items: "))
                            except:
                                print("Invalid input. Please enter an integer.")
                                continue
                            if selected_type == 1:
                                queue.type = "INT"

                            elif selected_type == 2:
                                queue.type = "Float"
                                
                            elif selected_type == 3:
                                queue.type = "String"
                                
                            else:
                                print("Invalid input")

                            while True:     
                                try:
                                    num_elements = int(input("Enter the number of elements you want to add: "))
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.\n")
                            for i in range(num_elements):
                                if queue.type == "INT":
                                    while True:
                                        try:
                                            item = int(input(f"Enter element {i + 1}: "))
                                        except:
                                            print("Invalid input. Please enter an integer.")
                                        queue.put(item)
                                        break

                                elif queue.type == "Float":
                                    while True:
                                        try:
                                            item = float(input(f"Enter element {i + 1}: "))
                                            queue.put(item)
                                        except:
                                            print("Invalid input. Please enter a float.")

                                elif queue.type == "String":
                                    item = input(f"Enter element {i + 1}: ")
                                    queue.put(item)
                            while True:
                                print("\nQueue Operations:")
                                print("1. Put Element")
                                print("2. Get Element")
                                print("3. Display Queue")
                                print("4. Back")
                                print("5. Exit")
                                try:
                                    operation = int(input("Enter your choice: "))
                                except:
                                    print("please enter a number.")
                                    continue
                            
                                if operation == 1:
                                    if queue.type == "INT":
                                        try:
                                            item = int(input("Enter item to Put: "))
                                        except:
                                            print("Invalid input. Please enter an integer.")
                                            continue
                                        queue.put(item)
                                    elif queue.type == "Float":
                                        try:
                                            item = float(input("Enter item to Put: "))
                                        except:
                                            print("Invalid input. Please enter a float.")
                                            continue
                                        queue.put(item)
                                    elif queue.type == "String":
                                        item = input("Enter item to Put: ")
                                        queue.put(item)
                                    print("Updated queue:", queue.items)
                                elif operation == 2:
                                    dequeued_item = queue.get()
                                    if dequeued_item is not None:
                                        print(f"Got item: {dequeued_item}")
                                        print("Updated queue:", queue.items)
                                    else:
                                        print("Queue is empty. Cannot dequeue.")

                                elif operation == 3:
                                    queue.display()
                                elif operation == 4:
                                    break
                                elif operation == 5:
                                    exit()
                                else:
                                    print("Invalid choice. Please try again.")
                    
                        elif select == 4:
                            break
                        elif select == 5:
                            exit()
                        else:
                            print("Invalid input")

        elif choice == 4:
            while True:
                print("\nWhat do you want?")
                print("1. Defenation")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter the choice: "))
                except:
                    print("Invalid input. Please enter an integer.")
                    continue
                if select == 1:
                    print("\nLinkedList Definition:\n\tA linked list is a linear data structure that consists of nodes where each node stores a data element and a reference (link) to the next node in the sequence.")
                elif select == 2:
                    print(""" a simple application of linked list is that is used in dynamic memory allocation
            and the code is :
            class Node:
        def __init__(self, data):
            self.data = data
            self.next = None

    class LinkedList:
        def __init__(self):
            self.head = None

        def push(self, data):
            new_node = Node(data)
            new_node.next = self.head
            self.head = new_node
    """)
                elif select == 3:
                    linked_list = LinkedList()
                    while True:
                        print("\nTypes of items: ")
                        print("1. Integer")
                        print("2. Float")
                        print("3. String")
                        try:
                            selected_type = int(input("Select the type of items: "))
                        except:
                            print("Invalid input. Please enter an integer.")
                            continue
                        
                        if selected_type == 1:
                            linked_list.type = "INT"
                            break
                        elif selected_type == 2:
                            linked_list.type = "Float"
                            break
                        elif selected_type == 3:
                            linked_list.type = "String"
                            break
                        else:
                            print("Invalid input")
                    try:
                        num_elements = int(input("Enter the number of elements you want to add: "))
                    except:
                        print("Invalid input. Please enter an integer.")
                        continue
                    for i in range(num_elements):
                        if linked_list.type == "INT":
                            while True:
                                try:
                                    item = int(input(f"Enter element {i + 1}: "))
                                    linked_list.insert(item)
                                    break
                                except:
                                    print("Invalid input. Please enter an integer.")
                        elif linked_list.type == "Float":
                            while True:
                                try:
                                    item = float(input(f"Enter element {i + 1}: "))
                                    linked_list.insert(item)
                                    break
                                except:
                                    print("Invalid input. Please enter a float.")
                        elif linked_list.type == "String":
                            item = input(f"Enter element {i + 1}: ")
                            linked_list.insert(item)

                    while True:
                        print("\nLinkedList Operations:")
                        print("1. Insert Element")
                        print("2. Delete Element")
                        print("3. Search Element")
                        print("4. Display LinkedList")
                        print("5. Back")
                        print("6. Exit")
                        try:
                            operation = int(input("Enter your choice: "))
                        except:
                            print("Invalid input. Please enter an integer.")
                            continue
                        if operation == 1:
                            if linked_list.type == "INT":
                                try:
                                    item = int(input("Enter item to insert: "))
                                except:
                                    print("Invalid input. Please enter an integer.")
                                    continue
                                linked_list.insert(item)
                                print("Updated linked list:")
                            elif linked_list.type == "Float":
                                try:
                                    item = float(input("Enter item to insert: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                                linked_list.insert(item)
                                print("Updated linked list:")
                            elif linked_list.type == "String":
                                item = input("Enter item to insert: ")
                                linked_list.insert(item)
                                print("Updated linked list:")
                            linked_list.display()
                        elif operation == 2:
                            if linked_list.type == "INT":
                                try:
                                    item = int(input("Enter item to delete: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                                linked_list.delete(item)
                            elif linked_list.type == "Float":
                                try:
                                    item = float(input("Enter item to delete: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                                linked_list.delete(item)
                            elif linked_list.type == "String":
                                item = input("Enter item to delete: ")
                                linked_list.delete(item)
                            print("Updated linked list:")
                            linked_list.display()
                        elif operation == 3:
                            if linked_list.type == "INT":
                                try:
                                    item = int(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter an integer.")
                                    continue
                            elif linked_list.type == "Float":
                                try:
                                    item = float(input("Enter item to search: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                            elif linked_list.type == "String":
                                item = input("Enter item to search: ")
                            
                            if linked_list.search(item):
                                print(f"Data {item} found in the linked list.")
                            else:
                                print(f"Data {item} not found in the linked list.")
                        elif operation == 4:
                            linked_list.display()
                        elif operation == 5:
                            break
                        elif operation == 6:
                            exit()
                        else:
                            print("Invalid choice. Please try again.")
                elif select == 4:
                    break
                elif select == 5:
                    exit()
                else:
                    print("Invalid input")
        
        elif choice == 5:
            while True:
                print("\nWhat do you want?")
                print("1. Defenation")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter the choice: "))
                except:
                    print("Invalid input. Please enter an integer.")
                    continue
                if select == 1:
                    print("\nTree Definition:\n\tA tree is a hierarchical data structure defined as a collection of nodes. Nodes represent value and nodes are connected by edges.")
                elif select == 2:
                    print("""\n A simple application of Tree is that it is used in file system
            and the example is :
            Root (Directory)
├── Documents (Directory)
│   ├── Report.doc (File)
│   ├── Presentation.ppt (File)
├── Pictures (Directory)
│   ├── Vacation.jpg (File)
│   ├── Family.jpg (File)
├── Music (Directory)
│   ├── Song1.mp3 (File)
│   ├── Song2.mp3 (File)

    """)
                
                elif select == 3:
                    while True:
                        print("Choose a tree:")
                        print("1. Binary Search Tree")
                        print("2. AVL Tree")
                        print("3. Red-Black Tree")
                        print("4. Back")
                        print("5. Exit")
                        choice = input("Enter your choice: ")

                        if choice == '1':
                            tree = BinarySearchTree()
                            while True:
                                print("\nChoose an option:")
                                print("1. Insert")
                                print("2. Search")
                                print("3. Inorder Traversal")
                                print("4. Back to Tree Selection")
                                print("5. exit")
                                choice = input("Enter your choice (1/2/3/4): ")

                                if choice == '1':
                                    try:
                                        data = int(input("Enter the data to insert: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    tree.insert(data)
                                elif choice == '2':
                                    try:
                                        data = int(input("Enter the data to search: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    result = tree.search(data)
                                    if result:
                                        print(f"{data} is found in the tree.")
                                    else:
                                        print(f"{data} is not found in the tree.")
                                elif choice == '3':
                                    print("Inorder Traversal:", tree.inorder_traversal())
                                elif choice == '4':
                                    break
                                elif choice == '5':
                                    exit()
                                else:
                                    print("Invalid choice. Please enter a valid option.")
                        elif choice == '2':
                            tree = AVLTree()
                            while True:
                                print("\nChoose an option:")
                                print("1. Insert")
                                print("2. Search")
                                print("3. Inorder Traversal")
                                print("4. Back to Tree Selection")
                                print("5. exit")
                                choice = input("Enter your choice (1/2/3/4): ")

                                if choice == '1':
                                    try:
                                        data = int(input("Enter the data to insert: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    tree.insert(data)
                                
                                elif choice == '2':
                                    break
                                elif choice == '3':
                                    break
                                elif choice == '4':
                                    break
                                elif choice == '5':
                                    exit()
                                else:
                                    print("Invalid choice. Please enter a valid option.")
                        elif choice == '3':
                            tree = RedBlackTree()
                            while True:
                                print("\nChoose an option:")
                                print("1. Insert")
                                print("2. Search")
                                print("3. Inorder Traversal")
                                print("4. Back to Tree Selection")
                                print("5. exit")
                                choice = input("Enter your choice (1/2/3/4): ")

                                if choice == '1':
                                    try:
                                        data = int(input("Enter the data to insert: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    tree.insert(data)
                                
                                elif choice == '2':
                                    try:
                                        data = int(input("Enter the data to search: "))
                                    except:
                                        print("Invalid input. Please enter an integer.")
                                        continue
                                    result = tree.search(data)
                                    if result:
                                        print(f"{data} is found in the tree.")
                                    else:
                                        print(f"{data} is not found in the tree.")
                                elif choice == '3':
                                    print("Inorder Traversal:", tree.inorder_traversal(tree.root))
                                elif choice == '4':
                                    break
                                elif choice == '5':
                                    exit()
                                else:
                                    print("Invalid choice. Please enter a valid option.")
                        elif choice == '5':
                            exit()
                        elif choice == '4':
                            break
                        else:
                            print("Invalid choice. Please enter a valid option.")
                            continue

                        

                elif select == 4:
                    break

                elif select == 5:
                    exit()

                else:
                    print("Invalid input.")  

        elif choice == 6:
            while True:
                print("\nWhat do you want?")
                print("1. Definition")
                print("2. Application")
                print("3. Practice")
                print("4. Back")
                print("5. Exit")
                try:
                    select = int(input("Enter your choice: "))
                except:
                    print("Invalid input. Please enter an integer.")
                    continue

                if select == 1:
                    print("\nGraph Definition:\n\t A graph is a data structure that consists of a set of nodes (vertices) connected by a set of edges. It is used to represent relationships or connections between various entities. Graphs can be directed (edges have a direction) or undirected (edges have no direction) and can be weighted (edges have associated values) or unweighted.")
                elif select == 2:
                    print("\nA simple application of a graph is finding mutual friends.")
                    print("Here's the code:")
                    print("""\nclass Graph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2, weight=None):
        if vertex1 in this.graph and vertex2 in self.graph:
            this.graph[vertex1].append((vertex2, weight))
            if not this.directed:
                this.graph[vertex2].append((vertex1, weight))

    def get_neighbors(self, vertex):
        return this.graph.get(vertex, [])

def find_mutual_friends(graph, user1, user2):
    user1_friends = set(graph.get_neighbors(user1))
    user2_friends = set(graph.get_neighbors(user2))
    mutual_friends = user1_friends.intersection(user2_friends)
    return mutual_friends

# Create an undirected unweighted social network graph
social_network = Graph()

# Add users (vertices)
social_network.add_vertex("Alice")
social_network.add_vertex("Bob")
social_network.add_vertex("Charlie")

# Add connections (edges)
social_network.add_edge("Alice", "Bob")
social_network.add_edge("Bob", "Charlie")

# Find mutual friends
user1 = "Alice"
user2 = "Charlie"
mutual_friends = find_mutual_friends(social_network, user1, user2)

print(f"Mutual friends between {user1} and {user2}: {mutual_friends}")
""")

                elif select == 3:
                    graph = Graph()
                    while True:
                        print("Graph Operations:")
                        print("1. Add Vertex")
                        print("2. Add Edge")
                        print("3. Display Graph")
                        print("4. Back")
                        print("5. Exit")
                        choice = input("Enter your choice: ")

                        if choice == '1':
                            vertex = input("Enter the vertex to add: ")
                            graph.add_vertex(vertex)
                        elif choice == '2':
                            vertex1 = input("Enter the first vertex: ")
                            vertex2 = input("Enter the second vertex: ")
                            is_weighted = input("Is the edge weighted? (yes/no): ").lower() == 'yes'
                            if is_weighted:
                                try:
                                    weight = float(input("Enter the edge weight: "))
                                except:
                                    print("Invalid input. Please enter a float.")
                                    continue
                                graph.add_edge(vertex1, vertex2, weight)
                            else:
                                graph.add_edge(vertex1, vertex2)
                        elif choice == '3':
                            print("Graph:")
                            graph.display()
                        elif choice == '4':
                            break
                        elif choice == '5':
                            print("Exiting the program.\n")
                            exit()
                        else:
                            print("Invalid choice. Please enter a valid option.")

        elif choice == 7:
            while True:
                try:
                    size = int(input("Enter the size of the hash table: "))
                    break
                except:
                    print("Invalid input. Please enter an integer.")
            hash_table = HashTable(size)

            while True:
                print("Hash Table Operations:")
                print("1. Insert Key-Value Pair")
                print("2. Retrieve Value by Key")
                print("3. Display")
                print("4. Back")
                print("5. Exit")
                choice = input("Enter your choice (1/2/3): ")

                if choice == '1':
                    key = input("Enter the key: ")
                    value = input("Enter the value: ")
                    hash_table.put(key, value)
                elif choice == '2':
                    key = input("Enter the key to retrieve: ")
                    value = hash_table.get(key)
                    if value is not None:
                        print(f"Value for key {key}: {value}")
                    else:
                        print(f"Key {key} not found.")
                elif choice == '3':
                    print("Exiting the program.")
                    for i in hash_table.table:
                        print(i)
                elif choice == '4':
                    break
                elif choice == '5':
                    print("Exiting the program.")
                    exit()
                else:
                    print("Invalid choice. Please enter a valid option.")
        elif choice == 8:
            exit()
        else:
            print("Invalid input")
if __name__ == "__main__":
    main()









