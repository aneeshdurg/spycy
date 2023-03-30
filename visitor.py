def visitor(ctx, f):
    if not f(ctx):
        return
    try:
        children = ctx.children
        if not children:
            children = []
    except AttributeError:
        children = []

    for c in children:
        visitor(c, f)


def hasType(ctx, type_):
    found_merge = False
    def helper(ctx):
        nonlocal found_merge
        if isinstance(ctx, type_):
            found_merge = True
            return False
        return True

    visitor(ctx, helper)
    return found_merge


def getType(ctx, type_):
    ctxs = []
    def helper(ctx):
        nonlocal ctxs
        if isinstance(ctx, type_):
            ctxs.append(ctx)
            return False
        return True

    visitor(ctx, helper)
    return ctxs
