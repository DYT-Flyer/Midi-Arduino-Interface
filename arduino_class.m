classdef arduino_class
    properties
        ard = arduino();
    end
    
    methods
        function turn_red_on(obj)
            writeDigitalPin(obj.ard,'D11',1);
        end
        
        function turn_red_off(obj)
            writeDigitalPin(obj.ard,'D11',0);
        end
    end
end