from SentiCR import  SentiCR

sentiment_analyzer=SentiCR()

#All examples are acutal code review comments from Go lang

sentences=["I'm not sure I entirely understand what you are saying. "+\
           "However, looking at file_linux_test.go I'm pretty sure an interface type would be easier for people to use.",
           "I think it always returns it as 0.",
           "If the steal does not commit, there's no need to clean up _p_'s runq. If it doesn't commit,"+\
             " runqsteal just won't update runqtail, so it won't matter what's in _p_.runq.",
           "Please change the subject: s:internal/syscall/windows:internal/syscall/windows/registry:",
           "I don't think the name Sockaddr is a good choice here, since it means something very different in "+\
           "the C world.  What do you think of SocketConnAddr instead?",
           "could we use sed here? "+\
            " https://go-review.googlesource.com/#/c/10112/1/src/syscall/mkall.sh "+\
            " it will make the location of the build tag consistent across files (always before the package statement).",
           "Is the implementation hiding here important? This would be simpler still as: "+\
          " typedef struct GoSeq {   uint8_t *buf;   size_t off;   size_t len;   size_t cap; } GoSeq;",
           "Make sure you test both ways, or a bug that made it always return false would cause the test to pass. "+\
        " assertTrue(Testpkg.Negate(false)); "+\
        " assertFalse(Testpkg.Negate(true)); +"\
        " If you want to use the assertEquals form, be sure the message makes clear what actually happened and " +\
        "what was expected (e.g. Negate(true) != false). "]

for sent in sentences:
    score=sentiment_analyzer.get_sentiment_polarity(sent)
    print(sent+"\n Score: "+str(score))