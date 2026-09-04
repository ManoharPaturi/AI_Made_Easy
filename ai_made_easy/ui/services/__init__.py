"""UI services: functionality objects (QObject-based, UI-widget-free).

They are the only things allowed to talk to core + canvas adapter; feature
widgets talk to services, never to core or OdenGraphQt directly.
"""
