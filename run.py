# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app, server
from pages import index, updates, howitworks

# Navbar docs: https://dash-bootstrap-components.opensource.faculty.ai/l/components/navbar
navbar = dbc.NavbarSimple(
    brand='Covid19 Modeling - Qatar',
    brand_href='/', 
    children=[
        dbc.NavItem(dcc.Link('How it works', href='/howitworks', className='nav-link')), 
        # dbc.NavItem(dcc.Link('Updates', href='/updates', className='nav-link')), 
    ],
    sticky='top',
    color='dark', 
    light=False, 
    dark=True,
    fluid=True,
)

# Footer docs:
# dbc.Container, dbc.Row, dbc.Col: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
# html.P: https://dash.plot.ly/dash-html-components
# fa (font awesome) : https://fontawesome.com/icons/github-square?style=brands
# mr (margin right) : https://getbootstrap.com/docs/4.3/utilities/spacing/
# className='lead' : https://getbootstrap.com/docs/4.3/content/typography/#lead
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span('Covid19-Qatar', className='mr-2'), 
                #    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:<you>@<provider>.com'), 
                    html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/A1337CBS/covid19MLPredictor'), 
                 #   html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/<you>/'), 
                 #   html.A(html.I(className='fab fa-twitter-square mr-1'), href='https://twitter.com/<you>'), 
                ], 
                className='lead'
            )
        )
    ),
    fluid=True,
)

# Layout docs:
# html.Div: https://dash.plot.ly/getting-started
# dcc.Location: https://dash.plot.ly/dash-core-components/location
# dbc.Container: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), 
    navbar, 
    dbc.Container(id='page-content', className='mt-4',fluid=True), 
    html.Hr(), 
    footer
])


# URL Routing for Multi-Page Apps: https://dash.plot.ly/urls
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index.layout
    elif pathname == '/howitworks':
        return howitworks.layout
    # elif pathname == '/updates':
    #     return updates.layout
    else:
        return dcc.Markdown('## Page not found')

# Run app server: https://dash.plot.ly/getting-started
if __name__ == '__main__':
    app.run_server(debug=True)