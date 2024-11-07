# player_potential.py

import pandas as pd

def get_top_players(Data, x_test, y_test, Predicted_Potential, top_n=10):
    """
    Create a DataFrame with player ID, actual potential, and predicted potential, and 
    return the top N players sorted by predicted potential.

    Parameters:
    - Data: DataFrame containing the player data (must include 'ID')
    - x_test: Test data indices
    - y_test: Actual potential values
    - predicted_potential: Predicted potential values (Combine_test)
    - top_n: Number of top players to return (default: 10)

    Returns:
    - top_players: DataFrame of top N players sorted by predicted potential
    """
    
    # Create a DataFrame with player ID, actual potential, and predicted potential
    result_df = pd.DataFrame({
    'Name': Data['Name'].iloc[x_test.index],
    'ID': Data['ID'].iloc[x_test.index],
    'Actual_Potential': y_test,
    'Predicted_Potential': Predicted_Potential
    })

    # Sort by predicted potential in descending order to get the top players
    top_players = result_df.sort_values(by='Predicted_Potential', ascending=False).head(top_n)

    return top_players
 