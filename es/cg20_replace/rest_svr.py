import json
import logging
import time
import ssl
import re
import asyncio
from aiohttp import ClientSession

import copy
import requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any, Union

from flask import Flask, request, jsonify, Response
from es.cg20_replace.logging_util import *
from elasticsearch import Elasticsearch

logger = logging.getLogger(__package__)

app = Flask("AIP")
es = Elasticsearch("192.168.242.214:9200,192.168.242.215:9200".split(","))
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
except Exception as exc:
    logger.error(exc)

class ESProxy(object):
    APTOTIC_BODY = {
        "apps": [
            {
                "entity": "c744de61-5a6a-4216-be66-34d614c4f461",
                "entity_type": 10
            }
        ],
        "body": ""
    }

    @staticmethod
    def format_data(data, req):
        """ Match the highlighting content

        Args:
            data (dict): elasticsearch result
            req (dict): user request data
        """

        query = req.get('query', {})
        ffilter = req.get('filter', {})
        keyword = req.get('keyword', {})
        key_terms = query.get('and', []) + query.get('or', []) + ffilter.get('and', []) + ffilter.get('or', [])
        nkey_terms = keyword.get('and', []) + keyword.get('or', [])
        hits = data.get('hits', {}).get('hits', [])
        for hit in hits:
            hg: dict = hit.get('highlight', {})
            if not hg:
                hit['highlight'] = {}
            for k in list(hg):
                v = hg[k]
                for i, _ in enumerate(v[:]):  # type:str
                    v[i] = _.replace("<em>", '').replace("</em>", '')
                    for nts in nkey_terms:
                        for t in nts.get('terms', []):
                            if str(t).lower() in str(v[i]).lower() and "{0}{1}{2}".format('<em>', str(t).lower(),
                                                                                          '</em>') not in str(
                                v[i]).lower():
                                v[i] = re.sub(r"({})".format(re.escape(t)), r'<em>\1</em>', v[i], flags=re.IGNORECASE)
                    for ks in key_terms:
                        for t in ks.get('terms', []):
                            if str(t).lower() in str(v[i]).lower() and ks.get('key') == k and "{0}{1}{2}".format(
                                    '<em>', str(t).lower(), '</em>') not in str(v[i]).lower():
                                v[i] = re.sub(r"({})".format(re.escape(t)), r'<em>\1</em>', v[i], flags=re.IGNORECASE)
                v[:] = [x for x in v if '<em>' in x]
                if not v:
                    del hg[k]

    @staticmethod
    async def fetch(t, session):
        result = es.search(index="tn_custom1_tp_unis*", doc_type="doc", body=json.dumps(t[1]))
        return t[0], result
        # url = 'http://{0}:{1}/tn_custom1_tp_unis*/doc/_search'.format("192.168.242.215", "9200")
        # async with session.get(url, data=json.dumps(t[1]),
        #                          headers={'Content-Type': 'application/json'}) as response:
        #     return t[0], await response.json()

    async def run(self, *args):
        tasks = []
        async with ClientSession() as session:
            for t in args:
                task = asyncio.ensure_future(ESProxy.fetch(t, session))
                tasks.append(task)

            self.responses = await asyncio.gather(*tasks)

    def search(self, t1, t2, t3):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(self.run(t1, t2, t3))
        loop.run_until_complete(future)


class PureSearchViewSet():
    """ search using the standard interface of ES

    """
    SEARCH_PERMISSION = 'search_file'

    def generate_paths(self, k: str, v: Union[str, list]):
        """ generate search paths

        Args:
            k (): [description]
            v ([type]): [description]

        Returns:
            [type]: [description]

        Examples:
            ['*', '/', '/*', '*/a', '*/a/*', 'a', '/a', 'a/b', 'b', 'b/', 'a/b/']
            ['/*', '/a/*', '/a/*', '/a/b/*', '/b/*', '/b/*', '/a/b/*', '*', '/*', '*/a', '*/a/*']
        """

        if k == 'path':
            if isinstance(v, str) and '*' not in v:
                v = v + '*' if v.endswith('/') else v + '/*'
                v = v if v.startswith('/') else '/' + v
            elif isinstance(v, list):
                v = list(map(lambda x: x + '*' if x.endswith('/') else x + '/*',
                             map(lambda x: x if x.startswith('/') else '/' + x,
                                 filter(lambda x: '*' not in x, v)))) + list(
                    filter(lambda x: '*' in x, v))
        return v

    def get_scope_permissions(self, scopes):
        """get scope's permissions

        Args:
            scope (Scope): scope
        """

        self.scope_permissions = []
        for _ in scopes:
            for meta in _.scopemetadatum_set.all():  # type: ScopeMetadatum
                self.scope_permissions.append((meta.key, meta.type, meta.value, _.name))

    def clean_empty_data(self, data: dict):
        bool_data = data.get('bool', {})  # type: dict
        tmp_bool_data = copy.copy(bool_data)
        for k, v in tmp_bool_data.items():
            if k == 'filter':
                if not v:
                    del data['bool'][k]
            else:
                v = filter(lambda x: x, v)
                p = list(v)
                if not p:
                    del data['bool'][k]
                else:
                    bool_data[k] = p

        if not data['bool']:
            del data['bool']

    def get_group_tree_terms(self, g, scope_name):
        scopes = g.scopes.filter(name=scope_name)
        self.get_scope_permissions(scopes)
        disperse_data = []
        self.group_tree_terms = {
            "bool": {
                "must": [],
            }
        }

        for term in self.scope_permissions:  # meta.key, meta.type, meta.value
            k, t, v = term[0], term[1], json.loads(term[2]).get('v')
            if v == []:  # do use `if not v` replace this
                v = ['']
            if isinstance(v, str):
                v = v.lower()
                if t == '=':
                    if '*' in v:
                        self.group_tree_terms['bool']['must'].append({'wildcard': {k: str(v).lower()}})
                    else:
                        self.group_tree_terms['bool']['must'].append({'term': {k: v}})
                elif t == 'in':
                    disperse_data.append({'key': k, 'terms': [v]})
                else:
                    pass

            elif isinstance(v, list):
                if t == '=':
                    for _ in v:
                        if '*' in _:
                            self.group_tree_terms['bool']['must'].append({'wildcard': {k: str(_).lower()}})
                        else:
                            self.group_tree_terms['bool']['must'].append({'term': {k: _}})
                elif t == 'in':
                    disperse_data.append({'key': k, 'terms': v})
                else:
                    pass
            else:
                pass
        self.group_tree_terms['bool']['must'] += ESDataFormat.generate_data(disperse_data, 'user')

    def get_role_terms(self, g):
        self.role_terms = []
        self.scope_names = set()
        parents = g.get_ancestors()
        for _ in g.scopes.values_list('name', flat=True):
            self.scope_names.add(_)
        for scope_name in self.scope_names:
            setattr(self, 'role_of_' + scope_name, {'bool': {'must': []}})
            self.role_terms.append(getattr(self, 'role_of_' + scope_name))
        for _ in (list(parents) + [g]):
            for scope_name in self.scope_names:
                self.get_group_tree_terms(_, scope_name)
                if self.group_tree_terms.get('bool').get('must'):
                    getattr(self, 'role_of_' + scope_name)['bool']['must'].append(self.group_tree_terms)

    def get_roles_terms(self, user=None):
        """

        Args:
            user (User):

        Returns:

        """
        self.roles_terms = {
            "bool": {
                "should": [],
            }
        }

        if not user:
            return

        groups = user.groups.all()
        for g in groups:
            self.get_role_terms(g)
            if not self.role_terms:
                self.roles_terms['bool']['should'].append({"source_id": ''})
            for role_term in self.role_terms:
                self.roles_terms['bool']['should'].append(role_term)

    def format_query_terms(self, data: dict, keywords: dict):
        """

        Args:
            data:
            keywords:

        Raises:

        Returns:
        """
        fdata = {
            "bool": {
                "must": [],
                "must_not": [],
            }
        }
        for oper, records in data.items():
            if oper == 'and':
                for record in records:
                    key = record.get('key', '')
                    terms = record.get('terms', [])
                    ranges = record.get('ranges', [])
                    for term in terms:
                        fdata['bool']['must'].append(ESDataFormat.create_wildcard_data(key, term))
                    for range in ranges:
                        fdata['bool']['must'].append(ESDataFormat.create_range_data(key, range))
            elif oper == 'or':
                fdata['bool']['must'] += ESDataFormat.generate_data(records, 'query')
            elif oper == 'not':
                for record in records:
                    key = record.get('key', '')
                    terms = record.get('terms', [])
                    ranges = record.get('ranges', [])
                    for term in terms:
                        fdata['bool']['must_not'].append(ESDataFormat.create_wildcard_data(key, term))
                    for range in ranges:
                        fdata['bool']['must_not'].append(ESDataFormat.create_range_data(key, range))

        keywords_fields = ["bucket", "fs_group", "fs_mode", "fs_mode", "metadata", "name", "ner", "owner",
                           "path", "source", "s3_tenant", "type", "perms", "command_result", "client_ip", "username",
                           "account", "db_name", "db_port", "protocol", "server_id", "cust_id", "cust_name",
                           "phone_num", "cert_type", "cert_number", "asr_txt"]

        keyword_should_data = {"bool": {"should": []}}
        keyword_must_data = {"bool": {"must": []}}
        keyword_not_data = {"bool": {"must_not": []}}
        for oper, records in keywords.items():
            if oper == 'and':
                for record in records:
                    terms = record.get('terms', [])
                    for term in terms:
                        tmp_kw_data = {'bool': {"should": []}}
                        for kf in keywords_fields:
                            tmp_kw_data['bool']['should'].append(ESDataFormat.create_wildcard_data(kf, term))
                        if tmp_kw_data['bool']['should']:
                            keyword_must_data['bool']['must'].append(tmp_kw_data)
                if keyword_must_data['bool']['must']:
                    fdata['bool']['must'].append(keyword_must_data)
            elif oper == 'or':
                for record in records:
                    terms = record.get('terms', [])
                    for term in terms:
                        for kf in keywords_fields:
                            keyword_should_data['bool']['should'].append(ESDataFormat.create_wildcard_data(kf, term))
                if keyword_should_data['bool']['should']:
                    fdata['bool']['must'].append(keyword_should_data)
            elif oper == 'not':
                for record in records:
                    terms = record.get('terms', [])
                    for term in terms:
                        for kf in keywords_fields:
                            keyword_not_data['bool']['must_not'].append(ESDataFormat.create_wildcard_data(kf, term))
                if keyword_not_data['bool']['must_not']:
                    fdata['bool']['must'].append(keyword_not_data)
        self.query_terms = fdata

    def format_filter_terms(self, data, pattern):
        fdata = {
            "bool": {
                "must": [],
                "must_not": [],
            }
        }
        for oper, records in data.items():
            if oper == 'or':
                if pattern == 'agg_time':
                    records = []
                elif pattern == 'agg_other':
                    records = [x for x in records if x.get('key') == 'last_modified']
                fdata['bool']['must'] += ESDataFormat.generate_data(records, 'filter')
            elif oper == 'and':
                for record in records:
                    key = record.get('key', '')
                    terms = record.get('terms', [])
                    ranges = record.get('ranges', [])
                    for term in terms:
                        fdata['bool']['must'].append(ESDataFormat.create_wildcard_data(key, term))
                    for range in ranges:
                        fdata['bool']['must'].append(ESDataFormat.create_range_data(key, range))
            elif oper == 'not':
                if pattern == 'agg_time':
                    records = []
                elif pattern == 'agg_other':
                    records = [x for x in records if x.get('key') == 'last_modified']
                for record in records:
                    key = record.get('key', '')
                    terms = record.get('terms', [])
                    ranges = record.get('ranges', [])
                    for term in terms:
                        fdata['bool']['must_not'].append(ESDataFormat.create_wildcard_data(key, term))
                    for range in ranges:
                        fdata['bool']['must_not'].append(ESDataFormat.create_range_data(key, range))

        self.filter_terms = fdata

    def get_aggs(self, field_name='last_modified'):
        time_k_t = {'last_1d': 1, 'last_1w': 7, 'last_1m': 30, 'last_1y': 365}
        rtime = {
            k: {
                'filter': {
                    'range': {
                        field_name: {
                            "gte":
                                int(time.mktime((datetime.now() -
                                                 timedelta(days=time_k_t[k])).timetuple()) * 1000)
                        }
                    }
                }
            }
            for k in list(time_k_t)
        }
        rtime['anytime'] = {
            'filter': {
                'range': {
                    field_name: {
                        'gte': 0
                    }
                }
            }
        }

        size_k_v = {
            '0-1_size': [0, 1024 * 1024],
            '1-5_size': [1024 * 1024, 1024 * 1024 * 5],
            '5-25_size': [1024 * 1024 * 5, 1024 * 1024 * 25],
            '25-100_size': [1024 * 1024 * 25, 1024 * 1024 * 100],
            "100-1_size": [1024 * 1024 * 100, 1024 * 1024 * 1024],
            '1+size': [1024 * 1024 * 1024]
        }
        size = {
            k: {
                'filter': {
                    'range': {
                        'size': {
                            "gte": size_k_v[k][0]
                        } if len(size_k_v[k]) == 1 else {
                            "gte": size_k_v[k][0],
                            "lte": size_k_v[k][-1]
                        }
                    }
                }
            }
            for k in list(size_k_v)
        }
        ftype = {'file_type': {"terms": {"field": "type.raw"}}}
        self.aggs = {**rtime, **size, **ftype}

    def get_high_light(self):
        self.high_light = {
            "require_field_match": True,
            "fields": {
                "*": {}
            }
        }

    def get_order(self, sort, order):
        if sort:
            if sort == 'name':
                sort = 'name.raw'
            self.sort_order = {sort: {"order": order}}
        else:
            self.sort_order = {}

    def get_pages(self, page_from, page_size):
        self.page_from = page_from
        self.page_size = page_size

    def combine_terms(self):
        self.clean_empty_data(self.filter_terms)
        self.query_terms['bool']['filter'] = self.filter_terms
        self.clean_empty_data(self.query_terms)
        self.clean_empty_data(self.roles_terms)
        terms = {"bool": {"must": [self.query_terms, self.roles_terms]}}
        self.clean_empty_data(terms)
        self.terms = {
            "aggregations": self.aggs,
            "sort": self.sort_order,
            "highlight": self.high_light,
            "from": self.page_from,
            "size": self.page_size
        }
        if terms:
            self.terms['query'] = terms
        return self.terms

    def cut_data(self, data, user, pattern='default'):
        query_data = data.get('query', {})
        filter_data = data.get('filter', {})
        sort = data.get('sort', None)
        order = data.get('order', 'desc')
        keyword_data = data.get('keyword', {})
        page_from = data.get('from', 0)
        page_size = data.get('pagesize', 20)
        self.format_query_terms(query_data, keyword_data)
        self.format_filter_terms(filter_data, pattern)
        self.get_roles_terms(user)
        self.get_aggs()
        self.get_high_light()
        self.get_order(sort, order)
        self.get_pages(page_from, page_size)
        return self.combine_terms()

    def create(self, request, *args, **kwargs):
        """[summary]

        Args:
            request ([type]): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        data = request.data
        # user: User = request.user
        user = None
        t1 = self.cut_data(data, user)
        t2 = self.cut_data(data, user, pattern='agg_time')
        t3 = self.cut_data(data, user, pattern='agg_other')
        esp = ESProxy()
        try:
            esp.search(('t1', t1), ('t2', t2), ('t3', t3))
        except Exception as ex:
            logger.error(str(ex))
            return Response({})
        r, tr, to = {}, {}, {}
        for _ in esp.responses:
            if _[0] == 't1':
                r = _[1]
            elif _[0] == 't2':
                tr = _[1]
            elif _[0] == 't3':
                to = _[1]
        # TDDO format data
        esp.format_data(r, data)
        aggs = tr.get('aggregations', {})
        time_keys = ['last_1d', 'last_1w', 'last_1m', 'last_1y', 'anytime']
        for key in list(aggs):
            if key in time_keys:
                r['aggregations'][key] = aggs[key]
        aggs = to.get('aggregations', {})
        for key in list(aggs):
            if key not in time_keys:
                r['aggregations'][key] = aggs[key]
        return Response(r)


class ESDataFormat(object):
    """ the data structure of ES is generated by user's permissions and input data
        just for logic or.
    """

    def __init__(self):
        pass

    @staticmethod
    def handle_disperse_data(data: list):
        """

        Args:
            data (list):  for example
                [
                    {"key": "name", 'terms': [1]},
                    {"key": "ip", 'terms': ['ip1', 'ip2', 'ip3'], 'ranges': [[0, 1], [0, 1]]},
                    {"key": "address", 'terms': ['a1', 'a2', 'a3'], 'ranges': [[0, 2], [2, 2]]},
                    {"key": "name", 'terms': [3, 4, 5], 'ranges': [[0, 0], [0, 0]]},
                    {"key": "", 'terms': [3, 4, 5], 'ranges': [[0, 0], [0, 0]]},
                ]

        Returns:
            [
                {'ranges': [[0, 0], [0, 0]], 'terms': [1, 3, 4, 5], 'key': 'name'}
                {'ranges': [[0, 1], [0, 1]], 'terms': ['ip1', 'ip2', 'ip3'], 'key': 'ip'}
                {'ranges': [[0, 2], [2, 2]], 'terms': ['a1', 'a2', 'a3'], 'key': 'address'}
                {'ranges': [[0, 0], [0, 0]], 'terms': [3, 4, 5], 'key': ''}
            ]

        """
        attributes, k_i, k_v = set(), {}, []
        for _ in data:
            attr = _.get('key')
            if attr not in attributes:
                k_v.append(_)
                attributes.add(attr)
                k_i[attr] = len(k_v) - 1
            else:
                i = k_i.get(attr)
                v = k_v[i]
                fields = list(_)
                fields.remove('key')
                for field in fields:
                    if field in list(v):
                        v[field] += _[field]  # type: list
                    else:
                        v[field] = _[field]
        return k_v

    @staticmethod
    def generate_data(data: list, source='user'):
        """

        Args:
            data (list):
                for example:
                [
                    {"key": "name", 'terms': [1]},
                    {"key": "ip", 'terms': ['ip1', 'ip2', 'ip3'], 'ranges': [[0, 1], [0, 1]]},
                    {"key": "address", 'terms': ['a1', 'a2', 'a3'], 'ranges': [[0, 2], [2, 2]]},
                    {"key": "name", 'terms': [3, 4, 5], 'ranges': [[0, 0], [0, 0]]},
                    {"key": "", 'terms': [3, 4, 5], 'ranges': [[0, 0], [0, 0]]},
                ]
            source (str):  user: data from user's permission

        Returns:
            for example:
            [
                {'bool': {'should': [{'wildcard': {'name': 1}}, {'wildcard': {'name': 3}}, {'wildcard': {'name': 4}}, {'wildcard': {'name': 5}}]}}
                {'bool': {'should': [{'wildcard': {'ip': 'ip1'}}, {'wildcard': {'ip': 'ip2'}}, {'wildcard': {'ip': 'ip3'}}]}}
                {'bool': {'should': [{'wildcard': {'address': 'a1'}}, {'wildcard': {'address': 'a2'}}, {'wildcard': {'address': 'a3'}}]}}
                {'bool': {'should': [{'wildcard': {'': 3}}, {'wildcard': {'': 4}}, {'wildcard': {'': 5}}]}}
            ]

        """
        data = ESDataFormat.handle_disperse_data(data)
        new_data = []
        for _ in data:
            tmp_data = {"bool": {"should": []}}
            k = _.get('key')
            for term in _.get('terms', []):
                if source == 'user':
                    if '*' not in term:
                        tmp_data['bool']['should'].append({"term": {k: term}})
                    else:
                        tmp_data['bool']['should'].append({"wildcard": {k: str(term).lower()}})
                else:
                    tmp_data['bool']['should'].append(ESDataFormat.create_wildcard_data(k, term))
            for r in _.get("ranges", []):
                tmp_data['bool']['should'].append(ESDataFormat.create_range_data(k, r))
            if tmp_data['bool']['should']:
                new_data.append(tmp_data)
        return new_data

    @staticmethod
    def create_wildcard_data(k, v):
        return {"wildcard": {k: '{0}{1}{0}'.format('*', str(v).lower())}}

    @staticmethod
    def create_range_data(k, v: list):
        return {'range': {k: {'gte': v[0]}}} if len(v) == 1 else {'range': {k: {'gte': v[0], 'lte': v[1]}}}


@app.route('/query', methods=['GET', 'POST'])
def query_data():
    print(request.json)
    data = request.json
    user = None
    ps = PureSearchViewSet()
    t1 = ps.cut_data(data, user)
    t2 = ps.cut_data(data, user, pattern='agg_time')
    t3 = ps.cut_data(data, user, pattern='agg_other')
    esp = ESProxy()
    print(t1)
    try:
        esp.search(('t1', t1), ('t2', t2), ('t3', t3))
    except Exception as ex:
        logger.error(str(ex))
        return Response({})
    r, tr, to = {}, {}, {}
    for _ in esp.responses:
        if _[0] == 't1':
            r = _[1]
        elif _[0] == 't2':
            tr = _[1]
        elif _[0] == 't3':
            to = _[1]
    # TDDO format data
    esp.format_data(r, data)
    aggs = tr.get('aggregations', {})
    time_keys = ['last_1d', 'last_1w', 'last_1m', 'last_1y', 'anytime']
    for key in list(aggs):
        if key in time_keys:
            r['aggregations'][key] = aggs[key]
    aggs = to.get('aggregations', {})
    for key in list(aggs):
        if key not in time_keys:
            r['aggregations'][key] = aggs[key]

    print(r)
    return Response(r)


if __name__ == '__main__':
    app.logger = get_logger("es", "DEBUG")
    app.run(
        host='0.0.0.0',
        port=9011,
        debug="DEBUG",
        use_reloader=False,
        threaded=True
    )
