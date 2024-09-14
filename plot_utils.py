import matplotlib.pyplot as plt

def Plot_actual_vs_predicted(y_train, y_test, Combine_train, Combine_test, train_color='blue', test_color='green'):
    """
    Function to plot actual vs predicted potential for both training and test datasets.
    
    Parameters:
    - y_train: Actual values for training data
    - y_test: Actual values for test data
    - Combine_train: Predicted values for training data
    - Combine_test: Predicted values for test data
    - train_color: Color for the training plot points (default is blue)
    - test_color: Color for the test plot points (default is green)
    """
    plt.figure(figsize=(12, 6))

    # Training plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, Combine_train, color=train_color, edgecolor='k')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Potential (Training)')
    plt.ylabel('Predicted Potential (Training)')
    plt.title('Actual vs Predicted (Training)')
    plt.grid(True)

    # Test plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, Combine_test, color=test_color, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Potential (Test)')
    plt.ylabel('Predicted Potential (Test)')
    plt.title('Actual vs Predicted (Test)')
    plt.grid(True)

    plt.suptitle('Actual vs Predicted Potential')
    plt.tight_layout()
    plt.show()
