# Data Professionals Job Market Insights

## Executive Summary
<p>
  This project presents a comprehensive analysis of the <strong>U.S. data job market (2023)</strong> to uncover actionable insights about 
  <strong>skills, salaries, and career progression</strong> for data professionals. 
  Using the Hugging Face dataset <code>lukebarousse/data_jobs</code>, the analysis explores thousands of job postings across multiple roles, 
  including <strong>Data Analyst</strong>, <strong>Data Scientist</strong>, <strong>Data Engineer</strong>, and 
  <strong>Machine Learning Engineer</strong>.
</p>

<p>
  Through systematic data preparation, cleaning, and visualization using 
  <strong>Python, Pandas, Matplotlib, and Seaborn</strong>, the study identifies:
</p>

<ul>
  <li>The <strong>most demanded and high-paying skills</strong> across junior and senior levels.</li>
  <li><strong>Skill-salary relationships</strong> that show where demand does not always match compensation.</li>
  <li><strong>Salary distributions</strong> by role and geography, revealing how experience and location shape pay.</li>
  <li>The <strong>most optimal skill sets</strong> that offer the best return on learning investment for data professionals.</li>
</ul>

<p>
  Overall, this analysis serves as a <strong>data-driven career guide</strong> for aspiring and experienced data professionals, emphasizing 
  the importance of <strong>Python, SQL, and cloud/AI-related skills</strong> for long-term career success.
</p>

## Overview

This project explores the US job posting market for data professionals (2023) using a dataset imported from the Hugging Face dataset `lukebarousse/data_jobs`. The analysis uses Python (Pandas, Matplotlib, Seaborn) to perform data preparation, exploratory data analysis, skills and salary analysis, and to recommend an optimal skill-set for different data roles.

## About dataset

Dataset source: https://huggingface.co/datasets/lukebarousse/data_jobs — job posting data for 2023 (US job postings subset was used). The dataset contains job titles, company, location, job posting date, salary information (when present), job attributes, and lists of skills/requirements extracted from postings.

## Tools used

- **Languages / Libraries:** Python 3.8+, Pandas, Matplotlib, Seaborn, datasets (Hugging Face), ast.  
- **Environment:** Jupyter Notebook.  
- **Version control:** Git, GitHub.


## The questions

The project answers the following questions:
1. What are the skills most in demand for different data roles?
2. How well do jobs and skills pay for different data roles?
3. How are salaries distributed across job titles and locations?
4. What are the most optimal skill-sets to target for someone entering the data job market?

## Data preparation and cleaning

This notebook handles the initial data ingestion, profiling, and cleaning. The primary objectives are to load the raw data from Hugging Face, inspect and correct key columns, filter the dataset to a relevant scope (US-based, full-time data jobs), and strategically impute missing data before exporting the cleaned file.

View my notebook with detailed steps here:
[2_ Skill_Demand.ipynb](1_Data_Prep_and_Cleaning.ipynb)

### Importing Files

Standard data analysis libraries (Pandas, Matplotlib, Seaborn) were imported. The ast library was included for parsing string-based lists, and datasets was used to load the data directly from the Hugging Face hub.

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from datasets import load_dataset

# importing dataset
dataset = load_dataset("lukebarousse/data_jobs")['train'].to_pandas()
```

### Data Cleaning

The data cleaning process involved several key steps. First, job_posted_date was converted to a datetime object, and the job_skills column, which was stored as a string, was parsed into a list. Redundant columns were dropped.

``` python
# data parsing
df['job_posted_date'] = pd.to_datetime(df.job_posted_date).dt.date
df['job_skills'] = df['job_skills'].apply(lambda skills: ast.literal_eval(skills) if pd.notna(skills) else skills)

# droping unnessary columns
df.drop(columns=
        ['job_type_skills', 'salary_hour_avg', 'salary_rate', 'search_location'], 
        inplace=True
)
df.head()
```

The dataset was then filtered to focus the analysis on "United States" job postings and "Full-time" roles. Finally, non-data-focused roles were excluded to refine the dataset to the target job titles.

``` python
# Focusing on US market only
df_US = df[df.job_country == 'United States'].copy()
df_US.sample()

# Focusing only on full time roles
df_US.job_schedule_type.value_counts().head()
df_US_filtered = df_US[df_US.job_schedule_type == 'Full-time'].copy()
df_US_filtered.job_schedule_type.count()

# Removing none data jobs
non_data_job = ['Cloud Engineer', 'Software Engineer']
df_US_filtered = df_US_filtered.drop(df_US_filtered[df_US_filtered['job_title_short'].isin(non_data_job)].index)

# sorting and reindex filtered datasets
df_US_filtered.sort_values(by='job_posted_date', ascending=True, inplace=True)
df_US_filtered.reset_index(inplace=True, drop=True)
```

### Filling out NA Values

A significant portion of salary_year_avg values were missing. These were imputed using a hierarchical median strategy to ensure statistical reliability.

A custom function fills missing salaries using a location-aware median approach. It first uses the median salary within each (job_location, job_title_short) group—only if that group has 10 or more valid records. Otherwise, it defaults to the overall median for that job title, ensuring accurate yet reliable imputation even for sparse data.

``` python
median_salaries = df_US_known.groupby('job_title_short')['salary_year_avg'].median()

# Step 1: finding medians by (job_location, job_title_short)
loc_title_medians = (
    df_US_known.groupby(['job_location', 'job_title_short'])
    .agg({'salary_year_avg': ['median', 'count']})
)
loc_title_medians.columns = ['median_salary', 'count']
loc_title_medians.sort_values(by='count', ascending=False, inplace=True)
loc_title_medians = loc_title_medians.reset_index()


# Step 2: function to fill out missing salaries
def fill_salary(row):
    subset = loc_title_medians[
        (loc_title_medians['job_location'] == row['job_location']) &
        (loc_title_medians['job_title_short'] == row['job_title_short'])
    ]

    if not subset.empty and subset.iloc[0]['count'] >= 10:
        return subset.iloc[0]['median_salary']
    else:
        return median_salaries[row['job_title_short']]

# Step 3: keeping the origin salary column
df_US_filled['original_salary_year_avg'] = df_US_filled['salary_year_avg']

# Step 4: fill out the na values
df_US_filled['salary_year_avg'] = df_US_filled.apply(
    lambda row: fill_salary(row) if pd.isna(row['salary_year_avg']) else row['salary_year_avg'],
    axis=1
)
```

# Analysis

Each Jupyter notebooks were designed to examine distinct dimensions of the data job market to answer the questions. My approach to each is summarized below:.

## 1. What are the skills most in demand for different data roles?

To identify the most in-demand skills, I un-nested the job_skills list using explode() to analyze each skill individually. I then counted the frequency of each skill per job title and calculated its percentage share relative to the total postings for that role. This normalized data was used to plot the top 5 skills for both junior and senior positions, revealing true demand within each category.

View my notebook with detailed steps here: [3_Job_Skills_Analysis.ipynb](3_Job_Skills_Analysis.ipynb)

### Visualize Data
```python
# plotting top 5 skills count for junior roles
fig, ax = plt.subplots(len(junior_roles), 1, figsize=(10,8), sharex=True)

for i, title in enumerate(junior_roles):
   data_series = top_skills[title]
   sns.barplot(data=data_series, x='% share', y='job_skills', ax=ax[i], hue='% share', palette='dark:b_r')

   ax[i].set_ylabel('')
   ax[i].set_title(title)
   ax[i].legend().remove()
   ax[i].set_xlabel('')
   ax[i].set_xlim(0, 86)

   for ind, v in enumerate(data_series['% share']):
      ax[i].text(v+0.25, ind, f'{round(v)}%', va='center')

fig.suptitle('Most Demanded job skills by job title', fontsize=18, fontweight='bold')
fig.tight_layout()
```
### Output
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="./plots/output-1.png" alt="Junior roles" width="90%" height="300">
      <br><strong>For Junior Roles</strong>
    </td>
    <td align="center" width="50%">
      <img src="./plots/output-2.png" alt="Senior roles" width="90%" height="300">
      <br><strong>For Senior Roles</strong>
    </td>
  </tr>
</table>


### Insights

<ul>
  <li>
    <strong>Python</strong> and <strong>SQL</strong> dominate the data domain, appearing as top skills across nearly all roles — with 
    <strong>Python</strong> demanded by up to <strong>85% of Senior Data Scientists</strong> and 
    <strong>78% of Machine Learning Engineers</strong>, and <strong>SQL</strong> required in 
    <strong>60–75%</strong> of analyst and engineering roles.
  </li>

  <li>
    <strong>Role specialization drives tool preference:</strong> 
    <strong>Analysts</strong> lean on <strong>Excel</strong> and <strong>Tableau</strong> for reporting 
    (up to <strong>49%</strong> and <strong>46%</strong> respectively), while 
    <strong>Engineers</strong> prioritize <strong>cloud</strong> and <strong>big data tools</strong> 
    like <strong>AWS (54%)</strong> and <strong>Spark (44%)</strong>, reflecting differing technical focuses.
  </li>

  <li>
    <strong>Machine Learning Engineers</strong> and <strong>Data Scientists</strong> emphasize 
    <strong>programming</strong> and <strong>modeling</strong> over visualization, with strong demand for 
    <strong>Python (78–85%)</strong>, <strong>R</strong>, and deep learning frameworks like 
    <strong>TensorFlow</strong> and <strong>PyTorch (~35%)</strong>.
  </li>

  <li>
    <strong>Seniority vs. Junior Roles:</strong> As professionals progress, the demand clearly shifts from 
    <strong>visualization and reporting tools</strong> (like <strong>Excel</strong> and <strong>Tableau</strong>) 
    toward <strong>programming</strong>, <strong>database</strong>, and <strong>cloud skills</strong> such as 
    <strong>Python</strong>, <strong>SQL</strong>, and <strong>AWS</strong>. 
    <strong>Senior roles</strong> expect deeper <strong>technical fluency</strong> and 
    <strong>system-level problem-solving</strong>, while <strong>junior roles</strong> focus more on 
    <strong>data interpretation</strong> and <strong>BI tools</strong>. 
    This highlights that advancing in data careers requires moving from 
    <strong>tool-based analysis</strong> to <strong>code-driven and scalable data solutions</strong>.
  </li>
</ul>

## 2. How well do jobs and skills pay for different data roles (eg: Data Analyst)?

To analyze skill value, I filtered the dataset for 'Data Analyst' roles. I then grouped by skill to aggregate both the total count (demand) and the mean salary (pay) for each skill. This allowed me to identify and plot the top 10 highest-paying and top 10 most in-demand skills for this specific role.

View my notebook with detailed steps here: [4_Salary_Analysis.ipynb](4_Salary_Analysis.ipynb)

### Visualize Data
``` python
# Select job title for the analysis
job_role = 'Data Analyst'

# Explode the skills list column
df_filtered = df[df.job_title_short == job_role]
df_exploded = df_filtered[df_filtered.job_skills.notna()].explode('job_skills', ignore_index=True)

# Highest Paid skills
most_paid_skills = (df_exploded.groupby('job_skills')['salary_year_avg']
                    .agg(['mean', 'count'])
                    .sort_values(by='mean', ascending=False)
                    .head(10)
                    .reset_index(names='job_skills')
                    .set_index('job_skills', drop=True)
)

# Most demanded skills
total_job_post = df_filtered['job_title_short'].count()
most_demand_skills = (df_exploded.groupby('job_skills')['salary_year_avg']
                      .agg(['count', 'mean'])
                      .sort_values(by='count', ascending=False)
                      .head(10)
                      .reset_index(names='job_skills')
                      .set_index('job_skills', drop=True)
)
most_demand_skills['%_share'] = 100*(most_demand_skills['count']/total_job_post)

# plotting top 10 demanded skills
fig, ax = plt.subplots(2,1, figsize=(10,8))
sns.barplot(data=most_demand_skills, x='%_share', y='job_skills', ax=ax[0])

# plotting top 10 paid skills
sns.barplot(data=most_paid_skills, x='mean', y='job_skills', ax=ax[1])
```

### Output
<div style="text-align: center;">
  <img src="./plots/output-3.png" alt="Most Demanded Skills">
</div>


### Insights for Data Analyst role
<ul>
  <li>
    <strong>High-demand vs. high-pay mismatch:</strong> While common tools like 
    <strong>SQL (51%)</strong>, <strong>Excel (41%)</strong>, and <strong>Tableau (29%)</strong> 
    dominate demand for Data Analysts, they are <strong>not the highest-paying skills</strong>, 
    showing that widely used tools often offer lower salary growth.
  </li>

  <li>
    <strong>Niche and emerging skills drive higher pay:</strong> Specialized frameworks such as 
    <strong>Hugging Face</strong>, <strong>MXNet</strong>, and <strong>Twilio</strong> 
    are among the <strong>most paid</strong>, reflecting that 
    <strong>AI, automation, and advanced programming expertise</strong> 
    command stronger compensation.
  </li>

  <li>
    <strong>Traditional vs. modern tool gap:</strong> Analysts relying mainly on 
    <strong>Excel</strong> and <strong>Tableau</strong> may face salary stagnation, while those 
    mastering <strong>machine learning frameworks</strong> or <strong>cloud-based tools</strong> 
    tend to access premium roles and higher pay scales.
  </li>

  <li>
    <strong>Career insight:</strong> To move beyond entry-level analyst pay, professionals should 
    gradually <strong>transition from reporting tools to technical and AI-focused skills</strong>, 
    blending <strong>data analysis</strong> with <strong>automation</strong> and 
    <strong>modeling capabilities</strong>.
  </li>
</ul>

## 3. How are salaries distributed by Job Title and Location?

To get an accurate view of salary distributions, I used the original salary data (before imputation) to preserve the true variance. I grouped the data by job_title_short and by job location to get the median salary for each role and for top 10 job location. Then sorted them from highest to lowest. Finally, I plotted box plots for both series to visualize the salary ranges, 25th/75th percentiles, medians, and outliers.

View my notebook with detailed steps here: [4_Salary_Analysis.ipynb](4_Salary_Analysis.ipynb)

### Visualize Data
``` python
# plotting salary distriubtion by job title
df_known = df[df['original_salary_year_avg'].notna()]
fig, ax = plt.subplots(len(job_titles), 1, figsize=(10, 8))

for i, job_title in enumerate(job_titles):
    temp = df_known[df_known.job_title_short == job_title]
    sns.boxplot(x='original_salary_year_avg', data=temp, ax=ax[i], orient='h')

fig.suptitle('Salary distribution for different job titles', ...)
```

``` python
# plotting salary distriubtion by job location
df_known = df_known[df_known['job_location'].notna()]
fig, ax = plt.subplots(len(job_loc), 1, figsize=(10, 8))

for i, job_loc in enumerate(job_loc):
    temp = df_known[df_known.job_location == job_loc]
    sns.boxplot(x='original_salary_year_avg', data=temp, ax=ax[i], orient='h')

fig.suptitle('Salary distribution for top 10 job location', ...)
```

### Output
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="./plots/output-4.png" alt="Distribution by Job Title" width="90%" height="300">
      <br><strong>Distribution by Job Title</strong>
    </td>
    <td align="center" width="50%">
      <img src="./plots/output-5.png" alt="Distribution by Location" width="90%" height="300">
      <br><strong>Distribution by Location</strong>
    </td>
  </tr>
</table>


### Insights
<ul>
  <li>
    <strong>Salary by Job Title:</strong> Senior roles like <strong>Senior Data Scientist</strong> and 
    <strong>Machine Learning Engineer</strong> show the highest median salaries, often exceeding 
    <strong>$150K+</strong>. In contrast, entry-level positions such as <strong>Data Analyst</strong> 
    and <strong>Business Analyst</strong> cluster around the <strong>$80K–$120K</strong> range, 
    showing a clear pay jump with experience and technical specialization.
  </li>

  <li>
    <strong>Location-based pay trends:</strong> Major tech hubs such as 
    <strong>San Francisco</strong>, <strong>New York</strong>, and <strong>Los Angeles</strong> consistently offer the <strong>highest compensation</strong>, driven by demand and cost of living, while regions like <strong>Tampa</strong> and <strong>Miami</strong> show comparatively lower salary ranges.
  </li>

  <li>
    <strong>Wider pay variability in senior and technical roles:</strong> Positions such as 
    <strong>Data Scientist</strong> and <strong>Data Engineer</strong> display broader 
    salary distributions, indicating high variability based on <strong>experience, project complexity,</strong> 
    and <strong>industry domain.</strong>
  </li>
</ul>

## 4. What are the most optimal skills to learn?

This analysis identifies the most optimal skills—those that are both high in demand and high-paying. I filtered the data for ‘Data Analyst’ roles, expanded the job_skills column, and grouped each skill by job_count and mean_salary. The top 10 in-demand skills were then plotted on a scatter chart showing their demand vs. average pay to reveal which skills offer the best return on investment.

View my notebook with detailed steps here: [5_Most_Optimal_Skills.ipynb](5_Most_Optimal_Skills.ipynb)

### Visualize Data
``` python
# selecting job role
job_role = 'Data Analyst'
df_filtered = df[df.job_title_short == job_role]

# exploding the skills list
df_exploded = df_filtered[df_filtered.job_skills.notna()].explode('job_skills', ignore_index=True)

df_scatter_data = (df_exploded.groupby('job_skills')
                  .agg({
                     'job_title':'count',
                     'salary_year_avg':'mean'
                  }).rename(columns={'salary_year_avg': 'mean_salary', 'job_title': 'job_count'})
                  .sort_values(by='job_count', ascending=False)
                  .head(10)
)
      
# Determine likelihood of job skill required in job posting
total_job_post = df_filtered['job_title_short'].count()
df_scatter_data['%_share'] = 100*(df_scatter_data['job_count']/total_job_post)

# plotting scatter plot
sns.scatterplot(data=df_scatter_data, x='%_share', y='mean_salary')

for i, skill in enumerate(df_scatter_data.index):
   xpoint = df_scatter_data.iloc[i,2]
   ypoint = df_scatter_data.iloc[i,1]
   plt.text(xpoint+0.35, ypoint, skill, fontsize=12, ha='left', va='center')

plt.title(f"Skill Demand vs Mean Salary for {job_role}", ...)
```

### Output
<div style="text-align: center;">
  <img src="./plots/output-6.png" alt="Most Demanded Skills">
</div>

### Insights

<ul>
  <li>
    <strong>Python and SQL remain the most optimal core skills</strong> across nearly all data roles — 
    offering both <strong>high demand</strong> and <strong>competitive pay</strong>. They consistently 
    appear on the upper right of the demand-salary charts, making them essential for any data career path.
  </li>

  <li>
    <strong>Strategic skill positioning by role:</strong>  
    For <strong>Data Scientists</strong> and <strong>Engineers</strong>, skills like <strong>Spark</strong>, 
    <strong>TensorFlow</strong>, and <strong>Kafka</strong> show moderate demand but offer 
    <strong>top-tier salaries</strong>, marking them as high-value differentiators.  
    Meanwhile, <strong>Analysts</strong> benefit most from blending widely used tools like 
    <strong>SQL</strong> and <strong>Tableau</strong> with an intermediate programming skill like 
    <strong>Python</strong> to unlock growth opportunities.
  </li>

  <li>
    <strong>Key takeaway:</strong> The most optimal career growth strategy is to build 
    a foundation in <strong>Python</strong> and <strong>SQL</strong>, then expand into 
    role-specific tools — <strong>Tableau/Power BI</strong> for Analysts, 
    <strong>ML frameworks</strong> for Scientists, and <strong>Cloud + Big Data</strong> 
    tools for Engineers and Senior roles.
  </li>
</ul>



## Conclusion
<p>
  The findings reveal a consistent trend across the data job landscape:
</p>

<ul>
  <li><strong>Python and SQL</strong> remain the foundational skills across nearly every data role.</li>
  <li><strong>Data Analysts</strong> benefit from strong <strong>reporting and visualization tools</strong>, while 
      <strong>Data Scientists</strong> and <strong>Data Engineers</strong> command higher pay through 
      <strong>programming, cloud, and machine learning expertise</strong>.</li>
  <li><strong>Location and seniority</strong> significantly influence salaries, with major tech hubs and senior technical roles 
      offering top-tier compensation.</li>
  <li>The <strong>most optimal career strategy</strong> is progressive learning — starting with analytical fundamentals, 
      then advancing into <strong>automation, cloud computing, and AI frameworks</strong>.</li>
</ul>

<p>
  In conclusion, the data job market rewards <strong>technical versatility</strong> and 
  <strong>continuous skill development</strong>. Professionals who combine strong technical execution with business understanding 
  are best positioned to thrive in this rapidly evolving field.
</p>

## How to Reproduce

1. **Clone the Repository**: Clone this repository to your local machine using:  
   ```bash
   git clone https://github.com/avinashchoudhary2004/Data-Professionals-Job-Market-Insights.git
   ```
2. **Run the Notebooks files**: Run 1_Data_Prep_and_Cleaning.ipynb to generate the cleaned dataset (data_cleaned_jobs.csv). Once the cleaned data file is created, proceed to run the remaining notebooks for further analysis and visualization.

3. **Explore & Customize**: You can easily modify parameters to generate new insights. For example, in 4_Salary_Analysis.ipynb, change the job_role variable (e.g., from "Data Analyst" to "Data Scientist") and re-run the notebook to produce updated visualizations and metrics for that role.

## Notes

- The dataset covers 2023 job postings — results may not reflect hiring trends beyond that period.
- To address widespread missing salaries, nulls were imputed using the median for that specific job title and location. Treat these imputed salaries as estimates.
