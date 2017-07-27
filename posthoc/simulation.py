import  mysql.connector
import  numpy as np
import  scipy.stats as stt
import pandas as pd
from pandas import DataFrame


db=mysql.connector.connect(user="bosu", password="password", host="localhost",
                           database="gerrit_android_simulation")

cur=db.cursor()

wilcoxh1=[]
pvalueh1=[]
effecth1=[]

chisqh2=[]
pvalueh2=[]
effecth2=[]

wilcoxh3=[]
pvalueh3=[]
effecth3=[]



for run in range(0, 100):

    print("Pass# "+ str(run))
    print("--------------------------------------")
    multi_sql="""TRUNCATE TABLE `randcomments`;
    
    TRUNCATE TABLE `randreviewcomments`;
    
    UPDATE inline_comments set simulation_score=sentiment_score;
    UPDATE review_comments set simulation_score=sentiment_score;
    
    INSERT INTO randcomments SELECT comment_id FROM `inline_comments` WHERE sentiment_score=0 ORDER by rand() LIMIT 0, 19494;
    
    INSERT INTO randcomments SELECT comment_id FROM `inline_comments` WHERE sentiment_score=-1 ORDER by rand() LIMIT 0, 7372;
    
    INSERT INTO `randreviewcomments` SELECT comments_id from review_comments WHERE sentiment_score=-1 ORDER by rand() limit 0, 4588;
    
    INSERT INTO `randreviewcomments` SELECT comments_id from review_comments WHERE sentiment_score=0 ORDER by rand() limit 0, 52497;
    
    -- INSERT INTO randcomments SELECT comment_id FROM `inline_comments` ORDER by rand() LIMIT 0, 25492;
    
    
    UPDATE inline_comments A INNER JOIN randcomments B on A.comment_id=B.comment_id set A.simulation_score = (-1* (A.sentiment_score + 1));
    
    UPDATE review_comments A INNER JOIN randreviewcomments B on A.comments_id=B.comments_id set A.simulation_score= (-1*(A.sentiment_score+1));
    
    DROP TABLE review_sentiment_simulation; 
    
    CREATE TABLE review_sentiment_simulation
    select request_id, count(*) as total_comments,
    SUM(CASE WHEN simulation_score=1 THEN 1 ELSE 0 END ) as positive_comments,
    SUM(CASE WHEN simulation_score=-1 THEN 1 ELSE 0 END ) as negative_comments,
    SUM(simulation_score) as sentiment_sum
    
    FROM (
    select request_id,author_id as author,simulation_score  from inline_comments
    UNION ALL
    select request_id,author, simulation_score  from review_comments WHERE sentiment_score is not NULL ) sentiment_calc
    group by request_id;
    
    ALTER TABLE `review_sentiment_simulation`
        ADD COLUMN `review_interval` INT NULL AFTER `sentiment_sum`,
        ADD COLUMN `status` INT NULL AFTER `review_interval`,
        ADD COLUMN `num_patches` INT NULL AFTER `status`,
        ADD COLUMN `code_churn` INT NULL AFTER `num_patches`,
        ADD COLUMN `has_negative` INT NULL AFTER `code_churn`,
        ADD COLUMN `has_positive` INT NULL AFTER `has_negative`,
        ADD COLUMN `sentiment_direction` INT NULL AFTER `has_positive`;
    
    UPDATE review_sentiment_simulation A inner join request_detail B on A.request_id=B.request_id
     set A.`status`= CASE WHEN B.`status`='ABANDONED' THEN 0 ELSE 1 END , A.review_interval =TIMESTAMPDIFF(SECOND,B.created,B.updated), A.num_patches=B.number_patches, A.code_churn= (B.insertions+B.deletions);
    
     UPDATE review_sentiment_simulation set has_negative = CASE WHEN negative_comments=0 THEN 0 ELSE 1 END;
    
     UPDATE review_sentiment_simulation set has_positive = CASE WHEN positive_comments=0 THEN 0 ELSE 1 END;
    
     UPDATE review_sentiment_simulation set sentiment_direction = CASE WHEN sentiment_sum=0 THEN 0 WHEN sentiment_sum<0 THEN -1 ELSE 1 END;
    
    ALTER TABLE `review_sentiment_simulation` ADD `num_files` INT NULL AFTER `sentiment_direction`;
    
    UPDATE review_sentiment_simulation A INNER JOIN (
    SELECT count(*) num_files,request_id,patchset_id
    FROM `patch_details`
    GROUP by request_id,patchset_id) B
    ON A.request_id=B.request_id and A.num_patches=B.patchset_id
    SET A.num_files=B.num_files;
    """

    for result in cur.execute(multi_sql, multi=True):
        pass

    numrows =cur.execute("SELECT request_id, review_interval, has_negative,status, num_patches  FROM `review_sentiment_simulation`")

    df = DataFrame(cur.fetchall())

    # Computing H1

    df1=df[df[2]==1]
    df2=df[df[2]==0]

    (w, p) = stt.mannwhitneyu(df1[1], df2[1])
    (rho,pval)=stt.spearmanr(df[1], df[2])

    wilcoxh1.append(w)
    pvalueh1.append(p)
    effecth1.append(rho)

    print ("(H1) Wilcox: "+str(w)+" p: "+ str(p)+ " corr: "+str(rho))

    #Computing H2
    (rho,pval)=stt.pearsonr(df[3], df[2])
    chi, p, degf, ar , =stt.chi2_contingency(pd.crosstab(df[3], df[2]))
    chisqh2.append(chi)
    pvalueh2.append(p)
    effecth2.append(rho)

    print ("(H2) Chisq: " + str(chi) + " p: " + str(p) + " corr: " + str(rho))

    #Computing H3
    (rho,pval)=stt.spearmanr(df[4], df[2])
    (w,p)=stt.mannwhitneyu(df1[4],df2[4])
    wilcoxh3.append(w)
    pvalueh3.append(p)
    effecth3.append(rho)

    print ("(H3) Wilcox: " + str(w) + " p: " + str(p) + " corr: " + str(rho))

training = open("simulation-mc-with-distribution.csv", 'w')
training.write("Run,Wilcox(H1),p(H1),Effect(H1),Chisq(H2),p(H2),Effect(H2),Wilcox(H3),p(H3),Effect(H3)\n")

for k in range(0, 100):
    training.write(str(k) + "," + str(wilcoxh1[k]) + "," + str(pvalueh1[k]) + "," + str(effecth1[k]) + "," +
                   str(chisqh2[k]) + "," + str(pvalueh2[k]) + "," + str(effecth2[k]) + "," +
                   str(wilcoxh3[k]) + "," + str(pvalueh3[k]) + "," + str(effecth3[k]) + "\n")
training.close()