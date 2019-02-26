
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

from pymongo import MongoClient

import pprint
import re
import json
import itertools

from datetime import datetime

from twitter_cascade_reconstruction import full_reconstruction

def get_url_domains(x,prefix='url: '):

    r = re.compile("(?:https?:)(?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n]+)")

    try:
        res = r.findall(x)
    except:
        res = []
    
    return(res)
    

def simulation_output_format_from_mongo_data_reddit(db='Jun19-train',start_date='2017-08-01',end_date='2017-08-05',
                                                    collection_name="Reddit_CVE_comments"):

    print('Extracting reddit data...')
    
    ############## Mongo queries ####################################
    #We store the comments and posts in two seperate collections so we have two different queries
    client = MongoClient()
    collection = client[db][collection_name]

    print('Querying mongo for comments...')
    mongo_json_data_comments = list(collection.find({"created_date":{"$gte":start_date + "T00:00:00Z",
                                                                     "$lt":end_date + "T00:00:00Z"}}))        
    ############## End mongo queries ####################################

    comments = pd.DataFrame(mongo_json_data_comments)

    
    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime', 'urlDomains','informationIDs']

    print('Extracting fields...')
    comments = comments.drop_duplicates('id_h')
    comments.rename(columns={'id_h':'nodeID','author_h':'nodeUserID',
                             'created_utc':'nodeTime','parent_id_h':'parentID','link_id_h':'rootID'}, inplace=True)

    comments.loc[:,'actionType']=['comment']*len(comments)
    comments.loc[:,'nodeID']=['t1_' + x for x in comments['nodeID']]
    comments.loc[:,'communityID'] = comments['subreddit_id']
    comments.loc[:,'informationIDs'] = [c['extension']['socialsim_keywords'] for i,c in comments.iterrows()]

    comments.loc[:,'urlDomains'] = comments["body_m"].apply(get_url_domains)
    

    ############## Mongo queries ####################################
    #extract posts data from mongo
    #We store the comments and posts in two seperate collections so we have two different queries
    collection_name = collection_name.replace("comments","posts")
    collection = client[db][collection_name]
    print('Querying mongo for posts...')
    mongo_json_data_posts = list(collection.find({"created_date":{"$gte":start_date + "T00:00:00Z",
                                                                  "$lt":end_date + "T00:00:00Z"}}))
    ############## End mongo queries ####################################

    posts = pd.DataFrame(mongo_json_data_posts)
    
    mongo_json_data = mongo_json_data_posts + mongo_json_data_comments
    
    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime', 'urlDomains','informationIDs']
    
    print('Extracting fields...')
    posts = posts.drop_duplicates('id_h')

    posts.rename(columns={'id_h':'nodeID','author_h':'nodeUserID','created_utc':'nodeTime'},
                 inplace=True)
    posts.loc[:,'nodeID']=['t3_' + x for x in posts['nodeID']]
    posts.loc[:,'rootID']=posts['nodeID']
    posts.loc[:,'parentID']=posts['nodeID']
    posts.loc[:,'actionType']=['post']*len(posts)
    posts.loc[:,'nodeAttributes']=[{'communityID':subredditID} for subredditID in posts['subreddit_id']]
    posts.loc[:,'communityID'] = posts['subreddit_id']
    posts.loc[:,'informationIDs'] = [p['extension']['socialsim_keywords'] for i,p in posts.iterrows()]
    posts.loc[:,'urlDomains'] = posts["selftext_m"].apply(get_url_domains)    

    reddit_data = pd.concat([comments[output_columns],posts[output_columns]],ignore_index=True)
    reddit_data = reddit_data.reset_index(drop=True)
        
    #remove broken portions
    reddit_data = reddit_data[reddit_data['parentID'].isin(list(set(reddit_data['nodeID'])))]
    reddit_data = reddit_data[reddit_data['rootID'].isin(list(set(reddit_data['nodeID'])))]

    print('Sorting...')
    reddit_data = reddit_data.sort_values('nodeTime')            

    #initialize info ID column with empty lists
    reddit_data['threadInfoIDs'] = [[] for i in range(len(reddit_data))]
    
    #for some reason having a non-object column in the dataframe messes up the assignment of lists to individual cell values
    #remove it temporarily and add back later
    nodeTimes = reddit_data['nodeTime']
    reddit_data = reddit_data[[c for c in reddit_data.columns if c != 'nodeTime']]

    #get children of node
    def get_children(nodeID):

        children = reddit_data[reddit_data['parentID'] == nodeID]['nodeID']
        children = children[children.values != nodeID]
        
        return(children)


    #all comments on a post/comment mentioning a unit of information are also assigned that unit of information
    def add_info_to_children(nodeID,list_info=[]):

        infos = list(reddit_data[reddit_data['nodeID'] == nodeID]['informationIDs'].values[0])

        list_info = list_info.copy()

        children = get_children(nodeID)
        
        if len(children) > 0:

            list_info += infos
    
            if len(list_info) > 0 and len(children) > 1:
                reddit_data.loc[children.index.values,'threadInfoIDs'] = [list_info for i in range(len(children))]
            elif len(list_info) > 0 and len(children) == 1:
                reddit_data.at[children.index[0],'threadInfoIDs'] = list_info

            for child in children.values:
                add_info_to_children(child,list_info)

    print('Adding information IDs to children...')
    #for each thread in data, propagate infromation IDs to children
    roots = reddit_data['rootID'].unique()
    for r,root in enumerate(roots):
        add_info_to_children(root)
        if r % 50 == 0:
            print('{}/{}'.format(r,len(roots)))
    
    reddit_data['nodeTime'] = nodeTimes

    reddit_data['informationIDs'] = reddit_data.apply(lambda x: list(set(x['informationIDs'] + x['threadInfoIDs'])),axis=1)
    reddit_data = reddit_data[reddit_data['informationIDs'].str.len() > 0]
    reddit_data = reddit_data.drop('threadInfoIDs',axis=1)

    print('Expanding events...')
    #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
    s = reddit_data.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'informationID'

    reddit_data = reddit_data.drop('informationIDs', axis=1).join(s).reset_index(drop=True)

    reddit_data = reddit_data.sort_values('nodeTime')
        
    return reddit_data,mongo_json_data
    

def simulation_output_format_from_mongo_data_twitter(db='Jun19-train',start_date='2017-08-01',end_date='2017-08-31',
                                                     collection_name="Twitter_CVE"):

    start_datetime = datetime.strptime(start_date,'%Y-%m-%d')
    end_datetime = datetime.strptime(end_date,'%Y-%m-%d')
    start_timestamp = str((start_datetime - datetime(1970,1,1)).total_seconds())[:-2] + '000'
    end_timestamp = str((end_datetime - datetime(1970,1,1)).total_seconds())[:-2] + '000'

    print('Timestamps:')
    print(start_timestamp,end_timestamp)
    
    ############## Mongo queries ####################################
    client = MongoClient()
    collection = client[db][collection_name]
    print('Querying mongo...')
    mongo_data_json = list(collection.find({"timestamp_ms":{"$gte":start_timestamp,
                                                            "$lt":end_timestamp}}))
    ############## End mongo queries ####################################

    mongo_data = pd.DataFrame(mongo_data_json)


    mongo_data = mongo_data.sort_values("timestamp_ms")

    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime', 'partialParentID','urlDomains','informationIDs']

    print('Extracting fields...')
    tweets = mongo_data.drop_duplicates('id_str_h')
    tweets.rename(columns={'id_str_h': 'nodeID',
                           'timestamp_ms': 'nodeTime'}, inplace=True)

    
    tweets.loc[:,'nodeTime'] = pd.to_datetime(tweets['nodeTime'],unit='ms')
    tweets.loc[:,'nodeTime'] = tweets['nodeTime'].apply(lambda x: datetime.strftime(x,'%Y-%m-%dT%H:%M:%SZ'))

    tweets.loc[:,'nodeUserID'] = tweets['user'].apply(lambda x: x['id_str_h'])

    tweets.loc[:,'is_reply'] = (tweets['in_reply_to_status_id_h'] != '') & (~tweets['in_reply_to_status_id_h'].isna())

    if 'retweeted_status.in_reply_to_status_id_h' not in tweets:
        tweets['retweeted_status.in_reply_to_status_id_h'] = ''
    if 'quoted_status.in_reply_to_status_id_h' not in tweets:
        tweets['quoted_status.in_reply_to_status_id_h'] = ''
    if 'quoted_status.is_quote_status' not in tweets:
        tweets['quoted_status.is_quote_status'] = False

    #keep track of specific types of reply chains (e.g. retweet of reply, retweet of quote of reply) because the parents and roots will be assigned differently
    tweets.loc[:,'is_retweet_of_reply'] = (~tweets['retweeted_status.in_reply_to_status_id_h'].isna()) & (~(tweets['retweeted_status.in_reply_to_status_id_h'] == ''))
    tweets.loc[:,'is_retweet_of_quote'] = (~tweets['retweeted_status'].isna()) & (~tweets['quoted_status'].isna()) & (tweets['quoted_status.in_reply_to_status_id_h'] == '')              
    tweets.loc[:,'is_retweet_of_quote_of_reply'] = (~tweets['retweeted_status'].isna()) & (~tweets['quoted_status'].isna()) & (~(tweets['quoted_status.in_reply_to_status_id_h'] == ''))
    tweets.loc[:,'is_retweet'] = (~tweets['retweeted_status'].isna()) & (~tweets['is_retweet_of_reply'])

    
    tweets.loc[:,'is_quote_of_reply'] = (~tweets['quoted_status.in_reply_to_status_id_h'].isna()) & (~(tweets['quoted_status.in_reply_to_status_id_h'] == '')) & (tweets['retweeted_status'].isna())
    tweets.loc[:,'is_quote_of_quote'] = (~tweets['quoted_status.is_quote_status'].isna()) & (tweets['quoted_status.is_quote_status'] == True) & (tweets['retweeted_status'].isna())
    tweets.loc[:,'is_quote'] = (~tweets['quoted_status'].isna()) & (~tweets['is_quote_of_reply']) & (~tweets['is_quote_of_quote']) & (tweets['retweeted_status'].isna())

    tweets.loc[:,'is_orig'] = (~tweets['is_reply']) & (~tweets['is_retweet']) & (~tweets['is_quote']) & (~tweets['is_quote_of_reply']) & (~tweets['is_quote_of_quote']) & (~tweets['is_retweet_of_reply'])
    
    to_concat = []

    replies = tweets[tweets['is_reply']]
    if len(replies) > 0:
        #for replies we know immediate parent but not root
        replies.loc[:,'actionType'] = 'reply'
        replies.loc[:,'parentID'] = tweets['in_reply_to_status_id_h']
        replies.loc[:,'rootID'] = '?'
        replies.loc[:,'partialParentID'] = tweets['in_reply_to_status_id_h']

        to_concat.append(replies)

    retweets = tweets[ (tweets['is_retweet']) & (~tweets['is_quote']) ]
    if len(retweets) > 0:
        #for retweets we know the root but not the immediate parent
        retweets.loc[:,'actionType'] = 'retweet'
        retweets.loc[:,'rootID'] = retweets['retweeted_status'].apply(lambda x: x['id_h'])
        retweets.loc[:,'parentID'] = '?'
        retweets.loc[:,'partialParentID'] = retweets['retweeted_status'].apply(lambda x: x['id_h'])

        to_concat.append(retweets)

    retweets_of_replies = tweets[ tweets['is_retweet_of_reply'] ]
    if len(retweets_of_replies) > 0:
        #for retweets of replies the "root" is actually the reply not the ultimate root
        #the parent of a retweet of a reply will be the reply or any retweet of the reply
        #the root can be retraced by following parents up the tree
        retweets_of_replies.loc[:,'parentID'] = '?'
        retweets_of_replies.loc[:,'rootID'] = '?'
        retweets_of_replies.loc[:,'partialParentID'] = retweets_of_replies['retweeted_status'].apply(lambda x: x['in_reply_to_status_id_h'])
        retweets_of_replies.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_replies)


    retweets_of_quotes = tweets[ tweets['is_retweet_of_quote'] ]
    if len(retweets_of_quotes) > 0:
        #for retweets of quotes we know the root (from the quoted status) but not the parent
        #the parent will be either the quote or any retweets of it
        retweets_of_quotes.loc[:,'parentID'] = '?'
        retweets_of_quotes.loc[:,'rootID'] = retweets_of_quotes['quoted_status'].apply(lambda x: x['id_h'])
        retweets_of_quotes.loc[:,'partialParentID'] = retweets_of_quotes['retweeted_status'].apply(lambda x: x['id_str_h'])
        retweets_of_quotes.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_quotes)


    retweets_of_quotes_of_replies = tweets[ tweets['is_retweet_of_quote_of_reply'] ]
    if len(retweets_of_quotes_of_replies) > 0:
        #for retweets of quotes of replies we don't know the root or the parent. the quoted status refers back to the reply not the final root
        #the parent will be either the quote or a retweet of the quote
        #we can find the root by tracking parents up the tree
        retweets_of_quotes_of_replies.loc[:,'parentID'] = '?'
        retweets_of_quotes_of_replies.loc[:,'rootID'] = '?'
        retweets_of_quotes_of_replies.loc[:,'partialParentID'] = retweets_of_quotes_of_replies['quoted_status'].apply(lambda x: x['id_str_h'])
        retweets_of_quotes_of_replies.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_quotes_of_replies)

                                                                                                                               
    quotes = tweets[tweets['is_quote']]
    if len(quotes) > 0:
        #for quotes we know the root but not the parent
        quotes.loc[:,'actionType'] = 'quote'
        quotes.loc[:,'rootID'] = quotes['quoted_status'].apply(lambda x: x['id_h'])
        quotes.loc[:,'parentID'] = '?'
        quotes.loc[:,'partialParentID'] = quotes['quoted_status'].apply(lambda x: x['id_h'])

        to_concat.append(quotes)

    quotes_of_replies = tweets[ tweets['is_quote_of_reply'] ]
    if len(quotes_of_replies) > 0:
        #for quotes of replies we don't know the root or the parent
        #the parent will be the reply or any retweets of the reply
        #the root can be tracked back using the parents in the tree
        quotes_of_replies.loc[:,'parentID'] = '?'
        quotes_of_replies.loc[:,'rootID'] = '?'
        quotes_of_replies.loc[:,'partialParentID'] = quotes_of_replies['quoted_status'].apply(lambda x: x['in_reply_to_status_id_h'])
        quotes_of_replies.loc[:,'actionType'] = 'quote'

        to_concat.append(quotes_of_replies)


    quotes_of_quotes = tweets[ tweets['is_quote_of_quote'] ]
    if len(quotes_of_quotes) > 0:
        #for quotes of quotes we don't know the parent or the root
        #the parent will be the first quote or any retweets of it
        #the root can be traced back through the parent tree
        quotes_of_quotes.loc[:,'parentID'] = '?'
        quotes_of_quotes.loc[:,'rootID'] = '?'
        quotes_of_quotes.loc[:,'partialParentID'] = quotes_of_quotes['quoted_status'].apply(lambda x: x['quoted_status_id_str'])
        quotes_of_quotes.loc[:,'actionType'] = 'quote'

        to_concat.append(quotes_of_quotes)


    orig_tweets = tweets[tweets['is_orig']]
    if len(orig_tweets) > 0:
        #for original tweets assign parent and root to be itself
        orig_tweets.loc[:,'actionType'] = 'tweet'
        orig_tweets.loc[:,'parentID'] = orig_tweets['nodeID']
        orig_tweets.loc[:,'rootID'] = orig_tweets['nodeID']
        orig_tweets.loc[:,'partialParentID'] = orig_tweets['nodeID']
        to_concat.append(orig_tweets)
                                                                                                                               
    tweets = pd.concat(to_concat,ignore_index=True)

    def url_wrapper(urls):

        url_list = []
        for url in urls:
            u = get_url_domains(url['expanded_url_h'])
            if len(u) > 0:
                u = u[0]
            if u != []:
                url_list.append(u)

        return(url_list)

    tweets.loc[:,'urlDomains'] = tweets["entities"].apply(lambda x: x['urls']).apply(url_wrapper)
    tweets.loc[:,'informationIDs'] = tweets['extension'].apply(lambda x: x['socialsim_keywords'])
    
    tweets = tweets[output_columns]

    print('Sorting...')
    tweets = tweets.sort_values("nodeTime")

    print('Reconstructing cascades...')
    tweets = full_reconstruction(tweets)

    #initialize info ID column with empty lists
    tweets['threadInfoIDs'] = [[] for i in range(len(tweets))]

    tweets = tweets.reset_index(drop=True)

    #get children of node
    def get_children(nodeID):

        children = tweets[tweets['parentID'] == nodeID]['nodeID']
        children = children[children.values != nodeID]
        
        return(children)


    #all comments on a post/comment mentioning a unit of information are also assigned that unit of information
    def add_info_to_children(nodeID,list_info=[]):

        infos = list(tweets[tweets['nodeID'] == nodeID]['informationIDs'].values[0])

        list_info = list_info.copy()

        children = get_children(nodeID)
        
        if len(children) > 0:

            list_info += infos
    
            if len(list_info) > 0 and len(children) > 1:
                #assign parents information ID list to all children
                tweets.loc[children.index.values,'threadInfoIDs'] = [list_info for i in range(len(children))]
            elif len(list_info) > 0 and len(children) == 1:
                #assign parents information ID list to single child
                tweets.at[children.index[0],'threadInfoIDs'] = list_info

            for child in children.values:
                #navigate further down the tree
                add_info_to_children(child,list_info)


    print('Adding information IDs to children...')
    #for each thread in data, propagate infromation IDs to children
    roots = tweets['rootID'].unique()
    for r,root in enumerate(roots):
        if root in tweets['nodeID'].values:
            add_info_to_children(root)
            if r % 50 == 0:
                print('{}/{}'.format(r,len(roots)))

    tweets['informationIDs'] = tweets.apply(lambda x: list(set(x['informationIDs'] + x['threadInfoIDs'])),axis=1)
    tweets = tweets[tweets['informationIDs'].str.len() > 0]
    tweets = tweets.drop('threadInfoIDs',axis=1)

    print('Expanding events...')
    #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
    s = tweets.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'informationID'
    
    tweets = tweets.drop(['informationIDs','partialParentID'], axis=1).join(s).reset_index(drop=True)
    
    return tweets,mongo_data_json


github_text_fields = {"PushEvent":["commits","message_m"],
                      "PullRequestEvent":["pull_request","body_m"],
                      "IssuesEvent":["issue","body_m"],
                      "CreateEvent":["description_m"],
                      "PullRequestReviewCommentEvent":["comment","body_m"],
                      "ForkEvent":["forkee","description_m"],
                      "IssueCommentEvent":["comment","body_m"],
                      "CommitCommentEvent":["comment","body_m"]}


def get_text_field(row):

    if row['actionType'] not in github_text_fields.keys():
        return ''
    
    if row['actionType'] == 'PushEvent':
        text = ' '.join(c['message_m'] for c in row['payload']['commits'])
    else:
        text = row['payload']
        
        for f in github_text_fields[row['actionType']]:
            if f in text:
                text = text[f]
            else:
                text = ''
            
        return text
    
def simulation_output_format_from_mongo_data_github(db='Jun2019-train',start_date='2017-08-01-01',end_date='2017-08-31',
                                                    collection_name="Github_CVE"):

    print('Extraing GitHub data...')

    ############## Mongo queries ####################################
    #we are storing the github data in two collections, one for events that mention an info ID and one for
    #events on repos whose readme or description matches the information ID. we just concatenate all those
    #events here.
    client = MongoClient()

    print('Querying mongo...')
    collection = client[db][collection_name + '_events']
    mongo_data_json_events = list(collection.find({"created_at":{"$gte":start_date + "T00:00:00Z",
                                                          "$lt":end_date + "T00:00:00Z"}}))


    collection = client[db][collection_name + '_repos']
    mongo_data_json_repos = list(collection.find({"created_at":{"$gte":start_date + "T00:00:00Z",
                                                          "$lt":end_date + "T00:00:00Z"}}))


    #concatenate two types of events
    mongo_data_json = mongo_data_json_events + mongo_data_json_repos
    ############## End mongo queries ####################################

    mongo_data = pd.DataFrame(mongo_data_json)


    print('Extracting fields...')
    output_columns = ['nodeID', 'nodeUserID', 'actionType', 'nodeTime','informationIDs','urlDomains','platform']

    if 'event' in mongo_data.columns:
        mongo_data.loc[:,'nodeTime'] = mongo_data['event'].apply(lambda x: x['created_at'])
        mongo_data.loc[:,'actionType'] = mongo_data['event'].apply(lambda x: x['type'])
        mongo_data.loc[:,'nodeUserID'] = mongo_data['event'].apply(lambda x: x['actor']['login_h'])
        mongo_data.loc[:,'nodeID'] = mongo_data['event'].apply(lambda x: x['repo']['name_h'])
    else:
        mongo_data.loc[:,'nodeUserID'] = mongo_data['actor'].apply(lambda x: x['login_h'])
        mongo_data.loc[:,'nodeID'] = mongo_data['repo'].apply(lambda x: x['name_h'])

        mongo_data.rename(columns={'created_at': 'nodeTime',
                                   'type':'actionType'}, inplace=True)

    mongo_data.loc[:,'platform'] = 'github'

    mongo_data.loc[:,'informationIDs'] = mongo_data['socialsim_details'].apply(lambda x: list(itertools.chain.from_iterable([m['extension']['socialsim_keywords'] for m in x])))

    mongo_data.loc[:,'urlDomains'] = mongo_data.apply(get_text_field,axis=1).apply(get_url_domains)

    events = mongo_data[output_columns]
    
    events = events[events.actionType.isin(['PullRequestEvent','IssuesEvent','CreateEvent','DeleteEvent','WatchEvent','ForkEvent',
                                            'PullRequestReviewCommentEvent','CommitCommentEvent','PushEvent','IssueCommentEvent'])]
    
    print('Expanding events...')    
    #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
    s = events.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'informationID'

    events = events.drop('informationIDs', axis=1).join(s).reset_index(drop=True)
    
    return events, mongo_data_json


def main():


    start_date = '2017-08-15'
    end_date = '2017-08-31'

    reddit_data,reddit_json_data = simulation_output_format_from_mongo_data_reddit(db='Jun19-train',
                                                                                   start_date=start_date,
                                                                                   end_date=end_date,
                                                                                   collection_name="Reddit_CVE_comments")

    print('Reddit output:')
    print(reddit_data)

    twitter_data,twitter_json_data = simulation_output_format_from_mongo_data_twitter(db='Jun19-train',
                                                                                      start_date=start_date,
                                                                                      end_date=end_date,
                                                                                      collection_name="Twitter_CVE")


    print('Twitter output:')
    print(twitter_data)

    
    github_data,github_json_data = simulation_output_format_from_mongo_data_github(db='Jun19-train',
                                                                                   start_date=start_date,
                                                                                   end_date=end_date,
                                                                                   collection_name="GitHub_CVE")

    print('Github output:')
    print(github_data)

    
    
    
if __name__ == "__main__":
    main()
