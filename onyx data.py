#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel("Onyx Data - DataDNA Dataset Challenge - Social Media Content Performance Dataset - June 2025.xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[9]:


null_summary = df.isnull().sum().sort_values(ascending=False)
print(null_summary)


# In[10]:



# Group and calculate average
content_category_stats = df.groupby("Content_Category")[["Engagement", "Views"]].mean().sort_values("Engagement", ascending=False).reset_index()

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=content_category_stats.melt(id_vars="Content_Category"), x="Content_Category", y="value", hue="variable")
plt.title("Average Engagement and Views by Content Category", fontsize=16)
plt.ylabel("Average Count")
plt.xlabel("Content Category")
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()


# In[11]:


# By Platform
df.groupby("Platform")[["Engagement", "Views"]].mean().sort_values("Engagement", ascending=False)

# By Post Type
df.groupby("Post_Type")[["Engagement", "Views"]].mean().sort_values("Engagement", ascending=False)


# In[12]:


df.groupby(["Region", "Content_Category"])[["Engagement", "Views"]].mean().unstack().fillna(0)


# In[13]:


# Convert post date if not already
df["Post_Date"] = pd.to_datetime(df["Post_Date"])
df["Day"] = df["Post_Date"].dt.day_name()

# By Hour
df.groupby("Post_Hour")["Engagement"].mean().plot(title="Engagement by Hour", figsize=(8,4))

# By Day
df.groupby("Day")["Engagement"].mean().plot(kind="bar", title="Engagement by Day", figsize=(8,4))


# In[26]:



region_perf = df.groupby("Region")[["Engagement", "Click_Through_Rate"]].mean().sort_values("Engagement", ascending=False).reset_index()

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Melt the DataFrame for seaborn
region_melt = region_perf.melt(id_vars="Region", value_vars=["Engagement", "Click_Through_Rate"])

# Plot
sns.barplot(data=region_melt, x="value", y="Region", hue="variable", palette="Set2")
plt.title("Average Engagement & Click-Through Rate by Region", fontsize=14)
plt.xlabel("Average Value")
plt.ylabel("Region")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


# In[15]:


df.groupby("Main_Hashtag")[["Impressions", "Clicks"]].mean().sort_values("Clicks", ascending=False).head(10)


# In[16]:


df.groupby("Region")[["Video_Views", "Live_Stream_Views"]].mean().sort_values("Video_Views", ascending=False).head(10)


# In[17]:


df.groupby(["Content_Category", "Post_Hour"])["Engagement"].mean().unstack().fillna(0)


# In[18]:


df.groupby("Content_Type")[["Engagement", "Views", "Impressions"]].mean()


# In[27]:



platform_engagement = df.groupby("Platform")["Engagement"].mean().reset_index().sort_values("Engagement", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=platform_engagement, x="Platform", y="Engagement", palette="viridis")
plt.title("Average Engagement by Platform")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[28]:


post_type_perf = df.groupby("Post_Type")[["Engagement", "Views", "Click_Through_Rate", "Clicks"]].mean().reset_index()
post_type_melt = post_type_perf.melt(id_vars="Post_Type")

plt.figure(figsize=(12,6))
sns.barplot(data=post_type_melt, x="Post_Type", y="value", hue="variable", palette="coolwarm")
plt.title("Performance Metrics by Post Type")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


# In[29]:


# Create flag for hashtag presence
df['Has_Hashtag'] = df['Main_Hashtag'].notnull()

hashtag_perf = df.groupby("Has_Hashtag")[["Engagement", "Views", "Clicks", "Click_Through_Rate"]].mean().reset_index()
hashtag_perf["Has_Hashtag"] = hashtag_perf["Has_Hashtag"].map({True: "With Hashtag", False: "Without Hashtag"})
hashtag_melt = hashtag_perf.melt(id_vars="Has_Hashtag")

plt.figure(figsize=(10,5))
sns.barplot(data=hashtag_melt, x="Has_Hashtag", y="value", hue="variable", palette="Set1")
plt.title("Performance Metrics: With vs Without Hashtag")
plt.ylabel("Average Value")
plt.xlabel("")
plt.tight_layout()
plt.show()


# In[22]:


category_region = df.groupby(["Region", "Content_Category"])["Engagement"].mean().unstack().fillna(0)
category_region.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20")
plt.title("Engagement by Content Category Across Regions")
plt.ylabel("Average Engagement")
plt.tight_layout()
plt.show()


# In[23]:


top_hashtags = df.groupby("Main_Hashtag")["Clicks"].sum().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=top_hashtags, x="Clicks", y="Main_Hashtag", palette="magma")
plt.title("Top 10 Hashtags by Clicks")
plt.xlabel("Total Clicks")
plt.ylabel("Hashtag")
plt.tight_layout()
plt.show()


# In[21]:


hourly_engagement = df.groupby("Post_Hour")["Engagement"].mean().reset_index()

plt.figure(figsize=(8,4))
sns.lineplot(data=hourly_engagement, x="Post_Hour", y="Engagement", marker="o")
plt.title("Engagement by Posting Hour")
plt.xticks(range(8, 20))
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


import plotly.express as px

country_video_views = df.groupby("Region")["Video_Views"].sum().reset_index()

fig = px.choropleth(country_video_views,
                    locations="Region",
                    locationmode="country names",
                    color="Video_Views",
                    color_continuous_scale="Plasma",
                    title="Total Video Views by Country")

fig.show()


# In[25]:


df.groupby("Region")[["Engagement", "Click_Through_Rate"]].mean().sort_values("Engagement", ascending=False)


# In[30]:



# --- PART 1: Engagement by Content Category & Engagement Level ---
cat_level = df.groupby(["Content_Category", "Engagement_Level"])["Engagement"].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=cat_level, x="Content_Category", y="Engagement", hue="Engagement_Level", palette="Spectral")
plt.title("Engagement by Content Category and Engagement Level")
plt.ylabel("Average Engagement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- PART 2: Engagement by Posting Hour ---
hourly = df.groupby("Post_Hour")["Engagement"].mean().reset_index()

plt.figure(figsize=(10,4))
sns.lineplot(data=hourly, x="Post_Hour", y="Engagement", marker='o', linewidth=2)
plt.title("Engagement by Hour of Day")
plt.xlabel("Post Hour")
plt.ylabel("Average Engagement")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PART 3: Heatmap of Content Category vs Hour ---
heat_data = df.pivot_table(values="Engagement", index="Content_Category", columns="Post_Hour", aggfunc="mean")

plt.figure(figsize=(12,6))
sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Engagement Heatmap: Content Category vs Posting Hour")
plt.xlabel("Post Hour")
plt.ylabel("Content Category")
plt.tight_layout()
plt.show()


# In[31]:



# Group by Content Type (Organic vs Promoted)
type_perf = df.groupby("Content_Type")[["Engagement", "Impressions", "Views", "Clicks", "Click_Through_Rate"]].mean().reset_index()

# Melt for Seaborn
type_melt = type_perf.melt(id_vars="Content_Type", value_vars=["Engagement", "Impressions", "Views", "Clicks", "Click_Through_Rate"])

# Plot
plt.figure(figsize=(12,6))
sns.barplot(data=type_melt, x="Content_Type", y="value", hue="variable", palette="Paired")
plt.title("Organic vs Promoted: Reach and Performance Metrics")
plt.ylabel("Average Value")
plt.xlabel("Content Type")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


# In[ ]:




