import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_insurance_data(n_samples=5000):
    """
    Generates a synthetic automobile insurance claims dataset.
    """
    
    # Generate common features
    months_as_customer = np.random.randint(0, 500, n_samples)
    age = np.random.randint(18, 75, n_samples)
    
    policy_state = np.random.choice(['OH', 'IL', 'IN'], n_samples)
    policy_csl = np.random.choice(['100/300', '250/500', '500/1000'], n_samples)
    policy_deductable = np.random.choice([500, 1000, 2000], n_samples)
    policy_annual_premium = np.random.uniform(500, 2500, n_samples).round(2)
    umbrella_limit = np.random.choice([0, 1000000, 2000000, 3000000, 4000000, 5000000], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.03, 0.02])
    
    insured_sex = np.random.choice(['MALE', 'FEMALE'], n_samples)
    insured_education_level = np.random.choice(['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College', 'JD'], n_samples)
    insured_occupation = np.random.choice(['machine-op-inspct', 'prof-specialty', 'tech-support', 'exec-managerial', 'sales', 'craft-repair', 'transport-moving', 'armed-forces', 'other-service', 'priv-house-serv'], n_samples)
    insured_hobbies = np.random.choice(['reading', 'exercise', 'paintball', 'chest', 'cross-fit', 'sleeping', 'camping', 'golf', 'movies', 'base-jumping', 'chess'], n_samples)
    insured_relationship = np.random.choice(['husband', 'wife', 'own-child', 'unmarried', 'other-relative', 'not-in-family'], n_samples)
    
    capital_gains = np.random.randint(0, 100000, n_samples)
    capital_gains = np.where(np.random.rand(n_samples) > 0.3, 0, capital_gains) # 70% have 0
    capital_loss = np.random.randint(-100000, 0, n_samples)
    capital_loss = np.where(np.random.rand(n_samples) > 0.3, 0, capital_loss)
    
    incident_type = np.random.choice(['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car', 'Vehicle Theft'], n_samples, p=[0.4, 0.4, 0.1, 0.1])
    collision_type = np.random.choice(['Side Collision', 'Rear Collision', 'Front Collision', '?'], n_samples)
    incident_severity = np.random.choice(['Minor Damage', 'Total Loss', 'Major Damage', 'Trivial Damage'], n_samples)
    authorities_contacted = np.random.choice(['Police', 'Fire', 'Ambulance', 'Other', 'None'], n_samples)
    incident_state = np.random.choice(['NY', 'SC', 'WV', 'VA', 'NC', 'PA', 'OH'], n_samples)
    incident_city = np.random.choice(['Columbus', 'Riverview', 'Arlington', 'Springfield', 'Hillsdale', 'Northbrook', 'Northbend'], n_samples)
    
    incident_hour_of_the_day = np.random.randint(0, 24, n_samples)
    number_of_vehicles_involved = np.random.randint(1, 5, n_samples)
    property_damage = np.random.choice(['YES', 'NO', '?'], n_samples)
    bodily_injuries = np.random.randint(0, 3, n_samples)
    witnesses = np.random.randint(0, 4, n_samples)
    police_report_available = np.random.choice(['YES', 'NO', '?'], n_samples)
    
    total_claim_amount = np.random.uniform(100, 120000, n_samples).round(2)
    injury_claim = (total_claim_amount * np.random.uniform(0.1, 0.3, n_samples)).round(2)
    property_claim = (total_claim_amount * np.random.uniform(0.1, 0.3, n_samples)).round(2)
    vehicle_claim = (total_claim_amount - injury_claim - property_claim).round(2)
    
    auto_make = np.random.choice(['Saab', 'Mercedes', 'Dodge', 'Chevrolet', 'Accura', 'Nissan', 'Audi', 'Toyota', 'Ford', 'Suburu', 'BMW', 'Jeep', 'Honda', 'Volkswagen'], n_samples)
    auto_model = np.random.choice(['Model A', 'Model B', 'Model C', 'Model D'], n_samples) # Simplified
    auto_year = np.random.randint(1995, 2023, n_samples)
    
    # Introduce Fraud Logics to make the dataset realistic
    # Genuine cases initially set to 0, Fraud to 1
    fraud_reported = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        prob_fraud = 0.05 # Base probability
        
        # High claim amount logic
        if total_claim_amount[i] > 80000:
            prob_fraud += 0.2
            
        # Severe incidents with NO police report logic
        if incident_severity[i] in ['Total Loss', 'Major Damage'] and police_report_available[i] == 'NO':
            prob_fraud += 0.3
            
        # Specific hobbies logic (based on historical real sets)
        if insured_hobbies[i] in ['chess', 'cross-fit']:
            prob_fraud += 0.15
            
        # Multiple vehicles but single accident
        if number_of_vehicles_involved[i] > 2 and incident_type[i] == 'Single Vehicle Collision':
            prob_fraud += 0.4
            
        # Age and short policy history
        if age[i] < 25 and months_as_customer[i] < 12 and total_claim_amount[i] > 50000:
            prob_fraud += 0.3
            
        # Trivial damage but high claim
        if incident_severity[i] == 'Trivial Damage' and total_claim_amount[i] > 20000:
            prob_fraud += 0.4
            
        # Cap probability at 0.95
        prob_fraud = min(prob_fraud, 0.95)
        
        # Determine actual fraud
        if np.random.rand() < prob_fraud:
            fraud_reported[i] = 1

    # Print class distribution roughly
    
    df = pd.DataFrame({
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_state': policy_state,
        'policy_csl': policy_csl,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'insured_hobbies': insured_hobbies,
        'insured_relationship': insured_relationship,
        'capital-gains': capital_gains,
        'capital-loss': capital_loss,
        'incident_type': incident_type,
        'collision_type': collision_type,
        'incident_severity': incident_severity,
        'authorities_contacted': authorities_contacted,
        'incident_state': incident_state,
        'incident_city': incident_city,
        'incident_hour_of_the_day': incident_hour_of_the_day,
        'number_of_vehicles_involved': number_of_vehicles_involved,
        'property_damage': property_damage,
        'bodily_injuries': bodily_injuries,
        'witnesses': witnesses,
        'police_report_available': police_report_available,
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'auto_make': auto_make,
        'auto_year': auto_year,
        'fraud_reported': fraud_reported
    })
    
    # Map fraud reported to Y/N
    df['fraud_reported'] = df['fraud_reported'].map({1: 'Y', 0: 'N'})
    
    return df

if __name__ == "__main__":
    print("Generating synthetic claims dataset...")
    df = generate_insurance_data(10000)
    df.to_csv("claims_data.csv", index=False)
    print(df['fraud_reported'].value_counts(normalize=True))
    print("Data saved to claims_data.csv.")
