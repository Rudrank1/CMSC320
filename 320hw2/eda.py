import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define questions
QUESTIONS = {
    1: "my sibling's nine year old child is poorly behaved. one day, my sibling dropped the child off on my doorstep without my consent and drove off, texting me that they needed me to watch their child for the day. the child complained of being hungry, so i gave them the leftovers i had. they said they didn't like them, but i told them that's all there was. they put them in the microwave until they burned, then told me they couldn't eat them and we would have to go out and get food. i told them that was what there was to eat, and if they ruined it they'd have to wait for their dad to get back. later, everyone yelled at me for 'starving' a child. am i a jerk?",
    2: "my parents want us to come out for their anniversary, and bought my sibling and i tickets on the same flight. my sibling has two kids - a 6 month old and a 5 year old. they are currently separated from their spouse so they would have to handle 2 children by themself on a 10 hour flight.\n\nthey call me up a week or so beforehand and ask me if i will be willing to help them take care of their kids on the flight, and something about taking shifts so we can both sleep. i tell them that i wasn't comfortable with that, but they say \"nephew loves you so much\" so we can work something out on the flight and hangs up.\n\nwhen i got to the airport, i chose to upgrade my seat so i could sit by myself and didn't mention it until we were on the plane. my family freaked and told me i abandoned my sibling. am i a jerk?",
    3: "one of my children wants to go to an expensive school to become a dentist. i told them i'd be fine paying for it. the other one wants to go to a similarly expensive school to become a teacher. i told them that i wouldn't pay for them schooling unless they picked a career path that will make at least 100,000 a year. am i a jerk?"
}

# Read and prepare data
def prepare_data():
    """Read and prepare the dataset with necessary transformations"""
    df = pd.read_csv('datasets/cleaned_combined_dataset.csv')
    
    # Filter for only male and female responses
    df = df[df['what bests represents your gender?'].isin(['Male', 'Female'])]
    
    # Filter for only Liberal and Conservative responses
    df = df[df['you could describe yourself as...'].isin(['Mildly conservative', 'Strongly conservative',
                                          'Mildly liberal', 'Strongly liberal'])]
    
    # Rename question columns to shorter names
    column_mapping = {
        QUESTIONS[1]: 'Q1',
        QUESTIONS[2]: 'Q2',
        QUESTIONS[3]: 'Q3'
    }
    df = df.rename(columns=column_mapping)
    
    # Keep only the three relevant questions and necessary columns
    columns_to_keep = [
        'you could describe yourself as...',
        'what bests represents your gender?',
        'Q1',
        'Q2',
        'Q3'
    ]
    df = df[columns_to_keep]
    
    # Save filtered dataset
    df.to_csv('datasets/filtered_dataset.csv', index=False)
    
    return df

def create_visualizations(df, question_num, question_text):
    """Create and save visualizations for a single question"""
    output_dir = os.path.join('figures', f'question_{question_num}')
    
    # 1. Distribution by political leaning (percentage)
    plt.figure(figsize=(10, 6))
    response_categories = pd.cut(df[f'Q{question_num}'], bins=[-np.inf, 0.5, 1.5, np.inf], 
                               labels=['Not a jerk', 'Mildly a jerk', 'Strongly a jerk'])
    cross_tab = pd.crosstab(df['you could describe yourself as...'], response_categories, normalize='index') * 100
    ax = cross_tab.plot(kind='bar', stacked=True)
    plt.title('Response Distribution by Political Leaning (%)')
    plt.ylabel('Question ' + str(question_num))
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_1.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Distribution by gender (percentage)
    plt.figure(figsize=(10, 6))
    cross_tab_gender = pd.crosstab(df['what bests represents your gender?'], response_categories, normalize='index') * 100
    ax = cross_tab_gender.plot(kind='bar', stacked=True)
    plt.title('Response Distribution by Gender (%)')
    plt.ylabel('Question ' + str(question_num))
    plt.xlabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_2.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Combined distribution (percentage)
    plt.figure(figsize=(12, 6))
    cross_tab_combined = pd.crosstab(
        [df['you could describe yourself as...'], df['what bests represents your gender?']],
        response_categories,
        normalize='index'
    ) * 100
    ax = cross_tab_combined.unstack().plot(kind='bar', stacked=True)
    plt.title('Response Distribution by Political Leaning and Gender (%)')
    plt.ylabel('Question ' + str(question_num))
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_3.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Mean response heatmap
    plt.figure(figsize=(10, 6))
    pivot_table = df.pivot_table(
        values=f'Q{question_num}', 
        index='you could describe yourself as...', 
        columns='what bests represents your gender?', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Mean Response Heatmap')
    plt.ylabel('Question ' + str(question_num))
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_4.png'), bbox_inches='tight')
    plt.close()

def perform_statistical_tests(df, question_num):
    """Perform statistical tests for both hypotheses"""
    # Test Hypothesis 1: Liberals vs Conservatives
    liberal_responses = df[df['you could describe yourself as...'].isin(['Mildly liberal', 'Strongly liberal'])][f'Q{question_num}']
    conservative_responses = df[df['you could describe yourself as...'].isin(['Mildly conservative', 'Strongly conservative'])][f'Q{question_num}']
    t_stat_pol, p_value_pol = stats.ttest_ind(liberal_responses, conservative_responses)
    
    # Test Hypothesis 2: Women vs Men
    female_responses = df[df['what bests represents your gender?'] == 'Female'][f'Q{question_num}']
    male_responses = df[df['what bests represents your gender?'] == 'Male'][f'Q{question_num}']
    t_stat_gen, p_value_gen = stats.ttest_ind(female_responses, male_responses)
    
    return {
        'political': {'t_stat': t_stat_pol, 'p_value': p_value_pol},
        'gender': {'t_stat': t_stat_gen, 'p_value': p_value_gen}
    }

def main():
    """Main function to run the analysis"""
    
    df = prepare_data()
    
    # Create output file for t-test results
    with open('misc/t_test_results.txt', 'w') as f:
        for question_num, question_text in QUESTIONS.items():
            f.write(f"\nAnalyzing Question {question_num}...\n")
            
            # Create visualizations
            create_visualizations(df, question_num, question_text)
            
            # Perform statistical tests
            results = perform_statistical_tests(df, question_num)
            
            # Write results to file
            f.write(f"\nStatistical Analysis for Question {question_num}:\n")
            f.write(f"\nHypothesis 1 (Political Leaning):\n")
            f.write(f"t-statistic: {results['political']['t_stat']:.4f}\n")
            f.write(f"p-value: {results['political']['p_value']:.4f}\n")
            f.write(f"\nHypothesis 2 (Gender):\n")
            f.write(f"t-statistic: {results['gender']['t_stat']:.4f}\n")
            f.write(f"p-value: {results['gender']['p_value']:.4f}\n")
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
