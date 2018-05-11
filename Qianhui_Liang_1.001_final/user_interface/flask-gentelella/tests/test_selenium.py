from test_base import urls


# this test uses selenium to check, for all pages of gentelella, that there
# are no severe entry in chromium logs.
# it is renamed selenium_test instead of test_selenium to be ignored by pytest
# (and therefore Travis CI). Rename it to test_selenium to make it a part
# of the test suite.
def selenium_test(selenium_client):
    for blueprint, pages in urls.items():
        for page in pages:
            selenium_client.get('http://127.0.0.1:5000' + blueprint + page)
            for entry in selenium_client.get_log('browser'):
                print(entry, blueprint, page)
                assert entry['level'] != 'SEVERE'
