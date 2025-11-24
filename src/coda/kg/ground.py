import gilda
import tqdm
import pandas as pd

exclude_list = ['Person', 'Death', 'Patients', 'gave', 'go', 'pm', 'PATIENT', 'bp', 'cold']

if __name__ == '__main__':
    df = pd.read_excel('../../data/ihme/PHMRC_VAI_redacted_free_text.xlsx', sheet_name='data')

    groundings = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row['open_response']):
            continue
        grounding = gilda.annotate(row['open_response'], namespaces=['MESH', 'DOID', 'HP'])
        groundings.append(grounding)
