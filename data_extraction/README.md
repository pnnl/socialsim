# Ground Truth Data Extraction

extract\_ground\_truth\_cp2.py demonstrates the approach for converting the raw JSON format for Reddit, Twitter, and GitHub to the simulation output schema for each platform.  The script is designed to query PNNL's mongo database, so you will have to modify the queries to interface with your individual data storage.  

The extraction process for each platform follows the follow steps:

1. Query a specific time period
2. Extract relevant fields from data
3. (For Twitter only) Assign roots and parents using the cascade reconstruction script
4. (Reddit and Twitter) Propagate any information IDs on parent posts/comments/tweets to all children of the post/comment/tweet
5. Duplicate events that are related to multiple information IDs. For example:
  * userA, tweetA, [CVE-2017-123, CVE-2014-456] will split into:
    * userA, tweetA, CVE-2017-123
    * userA, tweetA, CVE-2014-456