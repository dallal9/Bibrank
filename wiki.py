from mediawiki import MediaWiki
wikipedia = MediaWiki()
wikipedia.search('Variable latent semantic indexing')
p = wikipedia.page('Chess')
p.title
p.summary
p.categories
p.images
p.links
p.langlinks