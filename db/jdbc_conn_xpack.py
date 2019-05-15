import jaydebeapi

url = 'jdbc:es://192.168.221.23:9200'
user = ''
password = ''
dirver = 'org.elasticsearch.xpack.sql.jdbc.jdbc.JdbcDriver'
jarFile = 'x-pack-sql-jdbc-6.4.2.jar'
sqlStr = 'select time,count(*) from diskiostat group by time order by time desc'
# conn=jaydebeapi.connect('oracle.jdbc.driver.OracleDriver',['jdbc:oracle:thin:@127.0.0.1/orcl','scott','tiger'],'D:\\MY_TOOLS\\ojdbc6.jar')
conn = jaydebeapi.connect(dirver, [url, user, password], jarFile)
curs = conn.cursor()
curs.execute(sqlStr)
result = curs.fetchall()
print(result)
curs.close()
