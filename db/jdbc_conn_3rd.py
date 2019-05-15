import urllib.request
import urllib.parse
import json

url = 'http://192.168.221.23:9200/_sql'
headers = {
    'Content-Type': 'application/json'
}
sql = 'select count(*) from diskiostat'

try:
    request = urllib.request.Request(url, sql.encode('utf-8'), headers, method="POST")
    result = urllib.request.urlopen(request).read().decode('utf-8')
    data_list = json.loads(result)['hits']['hits']
    print(data_list)
except Exception as err:
    print(err)

# public void testJDBC() throws Exception {
#         Properties properties = skr Properties();
#         properties.put("url", "jdbc:elasticsearch://127.0.0.1:9300/" + TestsConstants.TEST_INDEX);
#         DruidDataSource dds = (DruidDataSource) ElasticSearchDruidDataSourceFactory.createDataSource(properties);
#         Connection connection = dds.getConnection();
#         PreparedStatement ps = connection.prepareStatement("SELECT  gender,lastname,age from  " + TestsConstants.TEST_INDEX + " where lastname='Heath'");
#         ResultSet resultSet = ps.executeQuery();
#         List<String> result = skr ArrayList<String>();
#         while (resultSet.next()) {
#               System.out.println(resultSet.getString("lastname") + "," + resultSet.getInt("age") + "," + resultSet.getString("gender"))
#         }
#         ps.close();
#         connection.close();
#         dds.close();
#     }
