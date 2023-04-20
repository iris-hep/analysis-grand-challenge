import particle
from anytree import NodeMixin, RenderTree, Node

class GenPartNode(NodeMixin):
    def __init__(self, name, genPart, parent=None, children=None):
        super(GenPartNode, self).__init__()
        self.name = name
        self.genPart = genPart
        self.parent = parent
        if children:
            self.children = children

def printTrees(particles):
    
    origins = []
    for genpart in particles:
        if genpart.genPartIdxMother==-1:
            
            # create origin node
            origin = GenPartNode(particle.Particle.from_pdgid(genpart.pdgId).name, genpart)
            origins.append(origin)

            # initialize lists/queues to keep track
            queue_node = []
            visited_genpart = []
            queue_genpart = []

            # add origin particle/node to queue/visited
            queue_node.append(origin)
            visited_genpart.append((genpart.pdgId,genpart.pt,genpart.eta,genpart.phi))
            queue_genpart.append(genpart)
            
            # loop through queue
            while queue_genpart:
            
                # grab top elements from queue
                g = queue_genpart.pop(0)
                n = queue_node.pop(0)

                # iterate through daughters
                for daughter in g.children:

                    # (should be) unique id for particle
                    daughter_tuple = (daughter.pdgId,daughter.pt,daughter.eta,daughter.phi)

                    # if we have not visited particle yet
                    if daughter_tuple not in visited_genpart:
                        
                        # add to queue
                        visited_genpart.append(daughter_tuple)
                        queue_genpart.append(daughter)

                        # create new node
                        node =  GenPartNode(particle.Particle.from_pdgid(daughter.pdgId).name, 
                                            daughter,
                                            parent = n)
                        
                        queue_node.append(node)
                                
        
    # printing trees
    for origin in origins:
        for pre, fill, node in RenderTree(origin):
            print("%s%s" % (pre, node.name))
            
    return