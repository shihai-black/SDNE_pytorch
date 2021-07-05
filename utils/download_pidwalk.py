%spark.pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

sql = """select cast(click_seq_len as string) as click_seq_len , pid_click_seq from  supply_chain_algorithm.tmp_embedding_pid_click_seq"""
sql_catid = """select product_id,cate_one_id from dw_dim.product_basic_info_df
where  date_id = '2021-05-23'
and channel = 'wholee'
and team = 'wholee'
and active_status = 1"""


s3_path = 's3://cf-supply/primary_profile/tmp/caojinlei/Ai_advertising/'
# sql1="""
# select click_seq_len,pid_click_seq,concat_ws(',',collect_list(cate_one_id)) as cate_click_seq from
# (select click_seq_len,pid_click_seq,cate_one_id from
# (select cast(click_seq_len as string) as click_seq_len , pid_click_seq,pid  from supply_chain_algorithm.tmp_embedding_pid_click_seq lateral view explode(split(pid_click_seq,',')) t as pid) a1
# left join
# (select product_id,cate_one_id from dw_dim.product_basic_info_df where  date_id = '2021-05-23' and active_status = 1)a2
# on a1.pid=a2.product_id
# )t
# group by click_seq_len,pid_click_seq
# """
spark = SparkSession.builder.appName('keyword').enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df_attr = spark.sql(sql_catid)
rdd_attr = df_attr.rdd.map(lambda x: (x[0].encode('utf8'), eval(x[1].encode('utf8'))))
attr_dict = rdd_attr.collectAsMap()
@F.udf(returnType=StringType())
def get_cate(pids):
    pid_list = pids.split(',')
    cat_list = []
    for pid in pid_list:
        if attr_dict.get(pid):
            cat = attr_dict[pid]
        else:
            cat = 0
        cat_list.append(cat)
    return str(cat_list)
df = spark.sql(sql1)
df = df.withColumn('cate_id_list',get_cate(df.pid_click_seq))
df.coalesce(1).write.mode('overwrite').csv(s3_path, sep=';')
