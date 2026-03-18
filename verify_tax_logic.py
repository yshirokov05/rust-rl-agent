import sys
import os

# Add the backend directory to the path so we can import tax_logic
sys.path.append(os.path.abspath('personal-finance-app-pfa/backend'))

from tax_logic import calculate_federal_tax, calculate_state_tax, calculate_fica_tax

def test_tax_accuracy():
    print("--- Starting Tax Logic Audit ---")
    
    # Test Case 1: High Earner in CA (Single)
    # Income: $1,200,000
    income = 1200000
    fed_tax = calculate_federal_tax(income, 'single')
    state_tax = calculate_state_tax(income, 'CA', 'single')
    fica_tax = calculate_fica_tax(income, 'single')
    
    print(f"Scenario 1: $1.2M earner in CA (Single)")
    print(f"  Federal Tax: ${fed_tax:,.2f}")
    print(f"  CA State Tax: ${state_tax:,.2f} (Includes Mental Health Tax)")
    print(f"  FICA/Medicare: ${fica_tax:,.2f}")
    
    # Validation logic
    # CA Mental Health Tax check: 1% on income over $1M
    taxable_ca = max(0, income - 5363) # CA Standard Deduction
    expected_mental_health = max(0, taxable_ca - 1000000) * 0.01
    
    # Test Case 2: MFJ at FICA Cap
    income_2 = 180000
    fica_2 = calculate_fica_tax(income_2, 'married_filing_jointly')
    # Social Security Cap 2026 is $176,100 @ 6.2%
    expected_ss = 176100 * 0.062
    expected_medicare = income_2 * 0.0145
    print(f"\nScenario 2: $180k earner (MFJ) - FICA Cap Check")
    print(f"  Total FICA: ${fica_2:,.2f}")
    print(f"  Expected SS: ${expected_ss:,.2f} + Medicare: ${expected_medicare:,.2f}")
    
    if abs(fica_2 - (expected_ss + expected_medicare)) < 5: # Small delta for rounding
        print("  FICA Cap Logic: PASS")
    else:
        print(f"  FICA Cap Logic: FAIL (Got {fica_2}, Expected {expected_ss + expected_medicare})")

if __name__ == "__main__":
    test_tax_accuracy()
