keyword_patterns = {
    "installation": [
        r"\binstall(?:ation|ing|ed|s)?\b", r"\bsetup(?:s)?\b", r"\bmount(?:ing|ed|s)?\b", r"\bconfigur(?:e|ation|ing|ed)?\b",
        r"\bfit(?:ting|tings|ted|s)?\b", r"\bassemb(?:ly|le|ling|led|ies)?\b", r"\banchor(?:ing|ed|s)?\b",
        r"\battach(?:ment|ments|ing|ed)?\b", r"\bplace(?:ment|d|s|ing)?\b", r"\bconstruct(?:ion|ing|ed|s)?\b",
        r"\bimplement(?:ation|ing|ed|s)?\b", r"\balign(?:ment|ing|ed|s)?\b", r"\bfasten(?:ing|er|ed|s)?\b",
        r"\bposition(?:ing|ed|s)?\b", r"\bsecure(?:ment|ing|d|s)?\b", r"\bconnect(?:ion|ing|ed|s)?\b",
        r"\baffix(?:ing|ed|s)?\b", r"\badapt(?:ation|ing|ed|s)?\b", r"\bretrofit(?:ting|ted|s)?\b",
        r"\bintegrat(?:e|ion|ing|ed|s)?\b", r"\bcalibrat(?:ion|e|ing|ed|s)?\b", r"\bwir(?:e|ing|ed|s)?\b",
        r"\bembed(?:ding|ded|s)?\b", r"\binstallment(?:s)?\b", r"\bharness(?:ing|ed|es)?\b", r"\bbolt(?:ing|ed|s)?\b",
        r"\bclip(?:ping|ped|s)?\b", r"\bdrill(?:ing|ed|s)?\b", r"\bslot(?:ting|ted|s)?\b", r"\bfastener(?:s)?\b",
        r"\bseal(?:ing|ed|s)?\b", r"\brivet(?:ing|ed|s)?\b"
    ],
    "repair": [
        r"\brepair(?:ing|ed|s)?\b", r"\bfix(?:ing|ed|es)?\b", r"\bissue(?:s|d)?\b", r"\bfault(?:y|s|ed)?\b",
        r"\breplac(?:e|ing|ed|ement|ements)?\b", r"\bdamage(?:d|s|ing)?\b", r"\brestore(?:d|ation|ing|s)?\b",
        r"\brectif(?:y|ication|ying|ied)?\b", r"\bpatch(?:ing|es|ed)?\b", r"\bservic(?:e|ing|es|ed)?\b",
        r"\btweak(?:ing|ed|s)?\b", r"\brefurbish(?:ing|ed|ment|ments)?\b", r"\boverhaul(?:ing|ed|s)?\b",
        r"\brejuvenate(?:d|ing)?\b", r"\bmalfunction(?:s|ing|ed)?\b", r"\brealign(?:ment|ing|ed)?\b",
        r"\bcalibrate(?:d|ing|s)?\b", r"\badjust(?:ment|ing|ed|s)?\b", r"\brectify(?:ing|ed)?\b"
    ],
    "qa": [
        r"\bquestion(?:s|ed|ing)?\b", r"\banswer(?:s|ed|ing)?\b", r"\bhow to\b", r"\bwhy\b", r"\bwhat\b",
        r"\bwhen\b", r"\bwhere\b", r"\bexplain(?:ation|ing|ed|s)?\b", r"\bclarify(?:ing|cation|ied|s)?\b",
        r"\bhelp(?:ing|ed|s)?\b", r"\btips?\b", r"\bguidance\b", r"\bexamples?\b", r"\bmanual(?:s)?\b",
        r"\bprocedure(?:s)?\b", r"\btutorial(?:s)?\b", r"\binstruction(?:s)?\b", r"\bguide(?:s|d)?\b"
    ],
    "maintenance": [
        r"\bmaintain(?:ing|ed|ance|s)?\b", r"\broutine(?:s)?\b", r"\bschedule(?:d|ing|s)?\b", r"\bcheck(?:ing|ed|s)?\b",
        r"\binspect(?:ion|ing|ed|ions)?\b", r"\bservic(?:e|ing|ed|es)?\b", r"\bupkeep(?:s)?\b", r"\bprevent(?:ive|ing|ed)?\b",
        r"\bclean(?:ing|ed|s)?\b", r"\blubricat(?:e|ion|ing|ed|s)?\b", r"\btune(?:up|ing|d|s)?\b", r"\badjust(?:ing|ed|s|ment)?\b",
        r"\breplace(?:ment|ments|ing|ed)?\b", r"\boptimize(?:d|ing|s)?\b", r"\bfluid(?: check|s)?\b", r"\boil(?:ing| filter)?\b",
        r"\bgrease(?:d|ing|s)?\b", r"\balign(?:ment|ing|ed|s)?\b", r"\brotate(?:d|ing|s)?\b"
    ],
    "troubleshooting": [
        r"\btroubleshoot(?:ing|ed|s)?\b", r"\bdiagnos(?:e|ing|is|ed|es)?\b", r"\berror(?:s)?\b", r"\bproblem(?:s|atic|ed)?\b",
        r"\bdebug(?:ging|ged|s)?\b", r"\bresolve(?:d|ing|s)?\b", r"\bcheck(?:ing|ed|s)?\b", r"\bdetermin(?:e|ing|ed|es)?\b",
        r"\bassess(?:ment|ing|ed|es)?\b", r"\bfailure(?:s)?\b", r"\bmisalignment(?:s)?\b", r"\broot cause(?:s)?\b",
        r"\bmalfunction(?:s|ing|ed)?\b", r"\banaly(?:z|s)(?:ing|ed|es|is)?\b", r"\balert(?:s)?\b", r"\bwarnings?\b",
        r"\bfault(?:y| code|s)?\b", r"\bscan(?:ning|ned|s)?\b"
    ],
    "upgrading": [
        r"\bupgrade(?:s|ing|d)?\b", r"\bretrofit(?:ting|ted|s)?\b", r"\bmodif(?:y|ication|ying|ied|s)?\b",
        r"\bcustomiz(?:e|ing|ed|ation|es)?\b", r"\btune(?:up|ing|d|s)?\b", r"\bperformance enhancement(?:s)?\b",
        r"\bimprov(?:e|ement|ing|ed|s)?\b", r"\bboost(?:ing|ed|s)?\b", r"\bmoderniz(?:e|ing|ed|ation|es)?\b",
        r"\bconvert(?:ing|ed|s)?\b", r"\baugment(?:ing|ed|ation|s)?\b", r"\bremodel(?:ing|ed|s)?\b",
        r"\boptimization(?:s)?\b", r"\btweak(?:ing|ed|s)?\b", r"\boverhaul(?:s|ed|ing)?\b"
    ],
    "fitting": [
        r"\bfit(?:ting|tings|ted|s)?\b", r"\battach(?:ment|ments|ing|ed)?\b", r"\bsecure(?:ment|ing|ed|s)?\b",
        r"\balign(?:ment|ing|ed|s)?\b", r"\bposition(?:ing|ed|s)?\b", r"\binsert(?:ing|ed|s)?\b",
        r"\bfasten(?:ing|er|ed|s)?\b", r"\bclip(?:ping|ped|s)?\b", r"\bseal(?:ing|ed|s)?\b",
        r"\banchor(?:ing|ed|s)?\b", r"\bsnug(?:ly)?\b", r"\badjust(?:ment|ing|ed|s)?\b", r"\brivet(?:ing|ed|s)?\b",
        r"\bbolt(?:ing|ed|s)?\b", r"\bslot(?:ted|ting|s)?\b"
    ],
    "parts": [
        r"\bengine(?:s)?\b", r"\bradiator(?:s)?\b", r"\bbattery(?:ies)?\b", r"\btire(?:s)?\b", r"\bwheel(?:s)?\b", r"\brim(?:s)?\b",
        r"\bbrake(?:s|pad(?:s)?)?\b", r"\bexhaust(?:s)?\b", r"\bsuspension(?:s)?\b", r"\bsteering\b", r"\bclutch(?:es)?\b",
        r"\bgear(?:box|shift)?\b", r"\balternator(?:s)?\b", r"\bfuel(?: tank| line| pump| filter)?\b", r"\bdashboard(?:s)?\b",
        r"\bmirror(?:s)?\b", r"\blight(?:s|ing|bulb(?:s)?)?\b", r"\bwiper(?:s| blades)?\b", r"\bbumper(?:s)?\b", r"\bhood(?:s)?\b",
        r"\bdoor(?:s)?\b", r"\btrunk(?:s)?\b", r"\bwindshield(?:s)?\b", r"\bair(?:bag| intake| filter)?\b", r"\bvalve(?:s)?\b",
        r"\bspark plug(?:s)?\b", r"\bchain(?:s)?\b", r"\bgasket(?:s)?\b", r"\bchassis\b", r"\bcylinder(?:s)?\b", r"\bpiston(?:s)?\b",
        r"\bdifferential(?:s)?\b", r"\baxle(?:s)?\b", r"\bcarburetor(?:s)?\b", r"\bturbocharger(?:s)?\b", r"\bdoor(?:s)?\b",
        r"\bseat(?:s)?\b", r"\bconsole(?:s)?\b", r"\bspare tire(?:s)?\b", r"\bhorn(?:s)?\b", r"\bspeaker(?:s)?\b",
        r"\bshock absorber(?:s)?\b", r"\bgrille(?:s)?\b", r"\bmuffler(?:s)?\b", r"\bhubcap(?:s)?\b", r"\bheadlight(?:s)?\b",
        r"\btaillight(?:s)?\b", r"\bwiring(?: harness)?\b"
    ]
}
