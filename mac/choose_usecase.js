// Ask which use case to start, printing the choice or failing on Cancel.
// Usage: osascript -l JavaScript choose_usecase.js <icon> <default> <usecase>...
ObjC.import("AppKit")

const DESCRIPTIONS = {
    clinical: "A clinician works up a death from clinical records",
    interview: "An interviewer asks a family about a death",
}

ObjC.registerSubclass({name: "RadioTarget", methods: {
    "pick:": {types: ["void", ["id"]], implementation: () => {}},
}})

function run(argv) {
    const [icon, preferred, ...usecases] = argv
    const app = $.NSApplication.sharedApplication
    app.setActivationPolicy($.NSApplicationActivationPolicyAccessory)
    const alert = $.NSAlert.alloc.init
    alert.messageText = "Start CODA"
    alert.informativeText = "Choose the kind of case for this session."
    alert.icon = $.NSImage.alloc.initWithContentsOfFile(icon)
    alert.addButtonWithTitle("Start")
    alert.addButtonWithTitle("Cancel")

    // Radio buttons with a description under each
    const target = $.RadioTarget.alloc.init
    const stack = $.NSStackView.alloc.initWithFrame($.NSMakeRect(0, 0, 340, 0))
    stack.orientation = $.NSUserInterfaceLayoutOrientationVertical
    stack.alignment = $.NSLayoutAttributeLeading
    stack.spacing = 4
    const radios = usecases.map(usecase => {
        const label = usecase[0].toUpperCase() + usecase.slice(1)
        const radio = $.NSButton.radioButtonWithTitleTargetAction(
            label, target, "pick:")
        radio.state = usecase === preferred ? $.NSControlStateValueOn
            : $.NSControlStateValueOff
        const text = $.NSTextField.labelWithString(DESCRIPTIONS[usecase] || "")
        text.font = $.NSFont.systemFontOfSize($.NSFont.smallSystemFontSize)
        text.textColor = $.NSColor.secondaryLabelColor
        const indent = $.NSStackView.alloc.init
        indent.edgeInsets = {top: 0, left: 20, bottom: 0, right: 0}
        indent.addViewInGravity(text, $.NSStackViewGravityLeading)
        stack.addViewInGravity(radio, $.NSStackViewGravityTop)
        stack.addViewInGravity(indent, $.NSStackViewGravityTop)
        stack.setCustomSpacingAfterView(12, indent)
        return radio
    })
    stack.layoutSubtreeIfNeeded
    stack.setFrameSize($.NSMakeSize(340, stack.fittingSize.height))
    alert.accessoryView = stack

    app.activateIgnoringOtherApps(true)
    if (alert.runModal !== $.NSAlertFirstButtonReturn) {
        throw new Error("Cancelled")
    }
    return usecases.find((_, i) => radios[i].state === $.NSControlStateValueOn)
}
