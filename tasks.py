from invoke import task


@task
def build_docs(ctx):
    ctx.run("sphinx-build docs/src build/docs", pty=True)
