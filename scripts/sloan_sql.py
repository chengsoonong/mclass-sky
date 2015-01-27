#!/usr/bin/python
""" >> sqlcl << command line SkyServer query tool
    
    Written by Tamas Budavari <budavari@jhu.edu>
    Modified by Alasdair Tran <github.com/alasdairtran> on 25 Jan 2015
    Usage: sqlcl [options] sqlfile(s)
    Currenty only works with one query at a time.
    Options:
        -s url	   : URL with the ASP interface (default: pha)
        -f fmt     : set output format (html,xml,csv - default: csv)
        -q query   : specify query on the command line
        -l         : skip first line of output with column names
        -o file    : output filename
        -v	       : verbose mode dumps settings in header
        -h	       : show this message

"""

sqlcl_formats = ['csv','xml','html']

astro_url='http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx'
public_url='http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx'

default_url=public_url
default_fmt='csv'
default_file='output.csv'

def sqlcl_usage(status, msg=''):
    "Error message and usage"
    print(__doc__)
    if msg:
        print('-- ERROR: %s' % msg)
    sys.exit(status)

def sqlcl_filtercomment(sql):
    "Get rid of comments starting with --"
    import os
    fsql = ''
    for line in sql.split('\n'):
        fsql += line.split('--')[0] + ' ' + os.linesep;
    return fsql

def sqlcl_query(sql,url=default_url,fmt=default_fmt):
    "Run query and return file object"
    import urllib.request, urllib.parse, urllib.error
    fsql = sqlcl_filtercomment(sql)
    params = urllib.parse.urlencode({'cmd': fsql, 'format': fmt})
    return urllib.request.urlopen(url+'?%s' % params)    

def sqlcl_write_header(ofp,pre,url,qry):
    import  time
    ofp.write('%s SOURCE: %s\n' % (pre,url))
    ofp.write('%s TIME: %s\n' % (pre,time.asctime()))    
    ofp.write('%s QUERY:\n' % pre)
    for l in qry.split('\n'):
        ofp.write('%s   %s\n' % (pre,l))
    
def sqlcl_main(argv):
    "Parse command line and do it..."
    import os, getopt, string
    
    queries = []
    url = os.getenv("SQLCLURL",default_url)
    fmt = default_fmt
    filename = default_file
    writefirst = 1
    verbose = 0
    
    # Parse command line
    try:
        optlist, args = getopt.getopt(argv[1:],'s:o:f:q:vlh?')
    except getopt.error as e:
        sqlcl_usage(1,e)
        
    for o,a in optlist:
        if   o=='-s': url = a
        elif o=='-f': fmt = a
        elif o=='-o': filename = a
        elif o=='-q': queries.append(a)
        elif o=='-l': writefirst = 0
        elif o=='-v': verbose += 1
        else: sqlcl_usage(0)
        
    if fmt not in sqlcl_formats:
        sqlcl_usage(1,'Wrong format!')

    # Enqueue queries in files
    for fname in args:
        try:
            queries.append(open(fname).read())
        except IOError as e:
            sqlcl_usage(1,e)

    # Run all queries sequentially
    for qry in queries:
        with open(os.path.normpath(filename), 'w') as f:
            ofp = sys.stdout
            if verbose:
                sqlcl_write_header(ofp,'#',url,qry)
                
            file = sqlcl_query(qry,url,fmt)
            
            # Output line by line (in case it's big)
            line = file.readline()
            if line.startswith(b"ERROR"): # SQL Statement Error -> stderr
                f = sys.stderr
            if writefirst:
                f.write(line.decode())
            line = file.readline()
            while line:
                f.write(line.decode())
                line = file.readline()


if __name__=='__main__':
    import sys
    sqlcl_main(sys.argv)





