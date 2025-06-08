def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    
    Args:
        arr (list): Sorted list to search in
        target: Element to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def test_binary_search():
    """Test function for binary search."""
    test_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    test_target = 7
    
    result = binary_search(test_array, test_target)
    
    if result != -1:
        print(f"Element {test_target} found at index {result}")
    else:
        print(f"Element {test_target} not found in the array")

if __name__ == "__main__":
    test_binary_search()