import pandas as pd

def drop_id_for_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the 'id' column from the DataFrame if it exists.
    Designed for use with sklearn's FunctionTransformer.
    """
    if 'id' in df.columns:
        return df.drop(columns=['id'])
    return df
