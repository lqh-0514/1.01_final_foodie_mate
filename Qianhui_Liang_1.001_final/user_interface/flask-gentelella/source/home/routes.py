from flask import Blueprint, render_template
from flask_login import login_required

blueprint = Blueprint(
    'home_blueprint',
    __name__,
    url_prefix='/home',
    template_folder='templates',
    static_folder='static'
)


@blueprint.route('/index')
@login_required
def index():

    return render_template('index.html',url_name="/static/images/img3.jpeg")


@blueprint.route('/index2')
@login_required
def index2():

    return render_template('index2.html')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    return render_template(template + '.html')
