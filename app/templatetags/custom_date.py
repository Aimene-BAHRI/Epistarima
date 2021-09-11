from django import template
from django.utils import formats
import datetime
register = template.Library()

@register.filter(expects_localtime=True, is_safe=False)
def custom_date(value, arg=None):
	if value in (None, ''):
		return ''

	if isinstance(value, str):
		try:
			value = datetime.datetime.strptime(value, '%Y-%m-%d')
		except ValueError as v:
			if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
				value = value[:-(len(v.args[0]) - 26)]
				t = datetime.datetime.strptime(value, '%Y-%m-%d')
			else:
				raise
	return t	


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)