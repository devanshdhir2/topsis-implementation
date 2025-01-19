import pandas as pd
import numpy as np
import sys
import os

def check_inputs(filename, weights, impacts, result_filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file'{filename}'does not exist")
    
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        raise Exception(f"Error reading input file: {str(e)}")
    
    if len(df.columns) < 3:
        raise ValueError("Input file must contain three or more columns")
    
    for col in df.columns[1:]:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            raise ValueError(f"Column'{col}'contains non-numeric values")
    
    weights = [float(w) for w in weights.split(',')]
    impacts = impacts.split(',')
    
    if not (len(weights) == len(impacts) == len(df.columns[1:])):
        raise ValueError("Number of weights, impacts, and columns must be the same")
    
    if not all(impact in ['+', '-'] for impact in impacts):
        raise ValueError("Impacts must be either '+' or '-'")
    
    return df, weights, impacts

def normalize_data(df):
    normalized_df = df.iloc[:, 1:].copy()
    
    for col in normalized_df.columns:
        denominator = np.sqrt(sum(normalized_df[col] ** 2))
        normalized_df[col] = normalized_df[col] / denominator
    
    return normalized_df

def calculate_topsis(normalized_df, weights, impacts):
    weighted_df = normalized_df * weights
    
    ideal_best = []
    ideal_worst = []
    
    for idx, col in enumerate(weighted_df.columns):
        if impacts[idx] == '+':
            ideal_best.append(weighted_df[col].max())
            ideal_worst.append(weighted_df[col].min())
        else:
            ideal_best.append(weighted_df[col].min())
            ideal_worst.append(weighted_df[col].max())
    
    s_best = np.sqrt(((weighted_df - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_df - ideal_worst) ** 2).sum(axis=1))
    
    topsis_score = s_worst / (s_best + s_worst)
    
    return topsis_score

def main():
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)
    
    try:
        input_file = sys.argv[1]
        weights = sys.argv[2]
        impacts = sys.argv[3]
        result_file = sys.argv[4]
        
        df, weights_list, impacts_list = check_inputs(input_file, weights, impacts, result_file)
        
        normalized_df = normalize_data(df)
        
        topsis_scores = calculate_topsis(normalized_df, weights_list, impacts_list)
        
        df['Topsis Score'] = topsis_scores
        df['Rank'] = df['Topsis Score'].rank(ascending=False)
        
        df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()