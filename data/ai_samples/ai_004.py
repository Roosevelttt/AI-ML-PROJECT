def process_data(data_list):
    """
    Process a list of data items.
    
    Args:
        data_list (list): List of data items to process
        
    Returns:
        list: Processed data items
    """
    processed_items = []
    
    for item in data_list:
        if isinstance(item, (int, float)):
            processed_item = item * 2
        elif isinstance(item, str):
            processed_item = item.upper()
        else:
            processed_item = str(item)
        
        processed_items.append(processed_item)
    
    return processed_items

def main():
    """Main execution function."""
    sample_data = [1, 2.5, "hello", True, None, "world"]
    result = process_data(sample_data)
    print("Processed data:", result)

if __name__ == "__main__":
    main()